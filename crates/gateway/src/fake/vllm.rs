//! A fake vLLM worker. It serves the endpoints the gateway touches, /metrics
//! and /v1/chat/completions, and records every chat request it receives. It
//! also plays vLLM's MooncakeConnector: GET /query is its bootstrap registry,
//! a request marked `do_remote_decode` stops at its one-token cap as a held
//! prefill does, and a request marked `do_remote_prefill` reports its whole
//! prompt as cached, as a pulled KV would make it. Statuses, headers and
//! bodies match the Go fake's, Go's `net/http` 404 and 405 answers included.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ariadne_decide::{RunId, parse_request_id};
use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::Response;
use bytes::Bytes;
use http::header::{ALLOW, CONTENT_TYPE, X_CONTENT_TYPE_OPTIONS};
use http::{HeaderMap, HeaderValue, Method, StatusCode};
use serde::Serialize;
use serde_json::{Value, json};

const JSON: HeaderValue = HeaderValue::from_static("application/json");

/// The knobs a test or a demo turns. `kv_usage`, `pool_blocks` and `waiting`
/// shape /metrics. `latency` delays every chat answer. A `status` other than
/// 200 makes chat answer that status with an OpenAI error body, and
/// `metrics_down` makes /metrics answer 503. `steps` is how many read-tool
/// calls a run makes before it submits, so a demo run has the doctor's
/// chained shape. `engine_id` is the engine the bootstrap registry reports.
/// `pull_latency` and `pull_status`, when set, replace `latency` and `status`
/// for a request marked `do_remote_prefill` only, as a destination stuck or
/// failing on a pull from an engine that is gone would behave.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub kv_usage: f64,
    pub pool_blocks: i64,
    pub waiting: f64,
    pub latency: Duration,
    pub status: StatusCode,
    pub metrics_down: bool,
    pub steps: usize,
    pub engine_id: String,
    pub pull_latency: Option<Duration>,
    pub pull_status: Option<StatusCode>,
}

/// One chat request as the worker saw it, and the bytes it answered with.
/// `response` is `None` while the request is being answered, and stays
/// `None` when the client left before the answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recorded {
    pub header: HeaderMap,
    pub body: Bytes,
    pub response: Option<Bytes>,
}

/// `last_body` keeps the previous chat body per run, so a later step of the
/// same run reports the shared prefix as cached tokens, as vLLM's prefix
/// cache would.
struct Inner {
    settings: Settings,
    requests: Vec<Recorded>,
    queries: usize,
    last_body: HashMap<RunId, Bytes>,
}

/// One fake vLLM worker. Clones share the same state, so a test keeps one
/// while the router serves from another.
#[derive(Clone)]
pub struct Worker {
    inner: Arc<Mutex<Inner>>,
}

impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}

impl Worker {
    /// Returns a Worker with an empty KV cache, the 4941-block pool of the
    /// reference deployment, no latency and a 200 status.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                settings: Settings {
                    kv_usage: 0.0,
                    pool_blocks: 4941,
                    waiting: 0.0,
                    latency: Duration::ZERO,
                    status: StatusCode::OK,
                    metrics_down: false,
                    steps: 0,
                    engine_id: "fake-engine".to_owned(),
                    pull_latency: None,
                    pull_status: None,
                },
                requests: Vec::new(),
                queries: 0,
                last_body: HashMap::new(),
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Edits the settings under the lock.
    pub fn set(&self, edit: impl FnOnce(&mut Settings)) {
        edit(&mut self.lock().settings);
    }

    /// How many times the bootstrap registry has been asked.
    pub fn queries(&self) -> usize {
        self.lock().queries
    }

    /// Returns a copy of every chat request so far, oldest first.
    pub fn requests(&self) -> Vec<Recorded> {
        self.lock().requests.clone()
    }

    /// Serves GET /metrics, GET /query and POST /v1/chat/completions.
    pub fn router(&self) -> Router {
        // One fallback instead of axum's method routing, whose 404 and 405
        // answers differ from Go's ServeMux in body and Allow header.
        Router::new().fallback(route).with_state(self.clone())
    }

    /// The Mooncake bootstrap registry: one engine at data-parallel rank 0.
    fn query(&self) -> Response {
        let engine = {
            let mut inner = self.lock();
            inner.queries += 1;
            inner.settings.engine_id.clone()
        };
        let mut body = go_json(&json!({"0": {"engine_id": engine}}));
        body.push('\n');
        json_response(StatusCode::OK, Bytes::from(body))
    }

    fn metrics(&self) -> Response {
        let settings = self.lock().settings.clone();
        if settings.metrics_down {
            return plain(StatusCode::SERVICE_UNAVAILABLE, "metrics unavailable");
        }
        let mut response = Response::new(Body::from(metrics_text(&settings)));
        response.headers_mut().insert(
            CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        );
        response
    }

    /// Records the request, waits `latency`, then answers. A client that
    /// leaves during the wait drops this future, so it gets no answer and
    /// its record keeps no response, as a cancelled upstream call would.
    async fn chat(&self, request: Request) -> Response {
        let (parts, body) = request.into_parts();
        let Ok(body) = axum::body::to_bytes(body, usize::MAX).await else {
            return plain(StatusCode::BAD_REQUEST, "read body");
        };
        let run = parts
            .headers
            .get("x-request-id")
            .and_then(|id| std::str::from_utf8(id.as_bytes()).ok())
            .and_then(parse_request_id)
            .map(|(run, _)| run);

        let (settings, cached, index) = {
            let mut inner = self.lock();
            let cached = run.map_or(0, |run| {
                let previous = inner.last_body.insert(run, body.clone());
                tokens(common_prefix(
                    previous.as_deref().unwrap_or_default(),
                    &body,
                ))
            });
            inner.requests.push(Recorded {
                header: parts.headers.clone(),
                body: body.clone(),
                response: None,
            });
            (inner.settings.clone(), cached, inner.requests.len() - 1)
        };

        let request = ChatRequest::parse(&body);
        let hop = request.transfer;
        let (mut latency, mut status) = (settings.latency, settings.status);
        if hop.do_remote_prefill {
            latency = settings.pull_latency.unwrap_or(latency);
            status = settings.pull_status.unwrap_or(status);
        }

        tokio::time::sleep(latency).await;

        let response = if status != StatusCode::OK {
            error_body(status.as_u16())
        } else if hop.do_remote_decode {
            held(&body)
        } else if hop.do_remote_prefill {
            completion(&request, body.len(), tokens(body.len()), settings.steps)
        } else {
            completion(&request, body.len(), cached, settings.steps)
        };
        if let Some(record) = self.lock().requests.get_mut(index) {
            record.response = Some(response.clone());
        }
        json_response(status, response)
    }
}

async fn route(State(worker): State<Worker>, request: Request) -> Response {
    let method = request.method();
    let read = method == Method::GET || method == Method::HEAD;
    match request.uri().path() {
        "/metrics" if read => worker.metrics(),
        "/query" if read => worker.query(),
        "/metrics" | "/query" => method_not_allowed("GET, HEAD"),
        "/v1/chat/completions" if method == Method::POST => worker.chat(request).await,
        "/v1/chat/completions" => method_not_allowed("POST"),
        _ => plain(StatusCode::NOT_FOUND, "404 page not found"),
    }
}

/// The subset of vLLM's exposition the fleet reads, with the pool size, the
/// waiting count and the KV usage filled in. The floats are written as Go's
/// `%g` writes them, because the Go fake's text is the reference.
fn metrics_text(s: &Settings) -> String {
    format!(
        "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{{block_size=\"16\",num_gpu_blocks=\"{}\"}} 1
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model_name=\"fake\"}} {}
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{model_name=\"fake\"}} {}
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{{model_name=\"fake\",le=\"0.1\"}} 40
vllm:time_to_first_token_seconds_bucket{{model_name=\"fake\",le=\"0.5\"}} 90
vllm:time_to_first_token_seconds_bucket{{model_name=\"fake\",le=\"1.0\"}} 99
vllm:time_to_first_token_seconds_bucket{{model_name=\"fake\",le=\"+Inf\"}} 100
vllm:time_to_first_token_seconds_count{{model_name=\"fake\"}} 100
vllm:time_to_first_token_seconds_sum{{model_name=\"fake\"}} 21.5
",
        s.pool_blocks,
        go_g(s.waiting),
        go_g(s.kv_usage),
    )
}

/// Formats `v` as Go's `fmt` `%g` does: the shortest digits that round-trip,
/// in exponent form when the exponent is below -4 or at least 6.
fn go_g(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_owned();
    }
    if v.is_infinite() {
        return if v > 0.0 { "+Inf" } else { "-Inf" }.to_owned();
    }
    let sign = if v.is_sign_negative() { "-" } else { "" };
    let scientific = format!("{:e}", v.abs());
    let Some((mantissa, exp)) = scientific.split_once('e') else {
        return scientific;
    };
    let digits = mantissa.replace('.', "");
    let exp: i32 = exp.parse().unwrap_or(0);
    if v == 0.0 {
        return format!("{sign}0");
    }
    if !(-4..6).contains(&exp) {
        let (first, rest) = digits.split_at(1);
        let point = if rest.is_empty() { "" } else { "." };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{first}{point}{rest}e{exp_sign}{:02}", exp.abs());
    }
    if exp < 0 {
        let zeros = "0".repeat(exp.unsigned_abs() as usize - 1);
        return format!("{sign}0.{zeros}{digits}");
    }
    let whole = exp as usize + 1;
    if digits.len() <= whole {
        format!("{sign}{digits:0<whole$}")
    } else {
        let (int, frac) = digits.split_at(whole);
        format!("{sign}{int}.{frac}")
    }
}

/// The part of a chat body the fake acts on. A field that is missing or of
/// the wrong type reads as its zero value, and a malformed body as all zero,
/// as Go's `json.Unmarshal` into a struct leaves them.
#[derive(Default)]
struct ChatRequest {
    model: String,
    tool_results: usize,
    transfer: TransferParams,
}

/// The part of `kv_transfer_params` the fake acts on.
#[derive(Clone, Copy, Default)]
struct TransferParams {
    do_remote_decode: bool,
    do_remote_prefill: bool,
}

impl ChatRequest {
    fn parse(body: &[u8]) -> Self {
        let Ok(request) = serde_json::from_slice::<Value>(body) else {
            return Self::default();
        };
        let flag = |name: &str| request["kv_transfer_params"][name].as_bool() == Some(true);
        Self {
            model: request["model"].as_str().unwrap_or_default().to_owned(),
            tool_results: request["messages"].as_array().map_or(0, |messages| {
                messages
                    .iter()
                    .filter(|m| m["role"].as_str() == Some("tool"))
                    .count()
            }),
            transfer: TransferParams {
                do_remote_decode: flag("do_remote_decode"),
                do_remote_prefill: flag("do_remote_prefill"),
            },
        }
    }
}

#[derive(Serialize)]
struct ChatCompletion<'a> {
    id: &'a str,
    object: &'a str,
    created: u64,
    model: &'a str,
    choices: [Choice<'a>; 1],
    usage: Usage,
}

#[derive(Serialize)]
struct Choice<'a> {
    index: u32,
    message: Message<'a>,
    finish_reason: &'a str,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: Option<&'a str>,
    tool_calls: Option<[ToolCall<'a>; 1]>,
}

#[derive(Serialize)]
struct ToolCall<'a> {
    id: String,
    r#type: &'a str,
    function: Function<'a>,
}

#[derive(Serialize, Clone, Copy)]
struct Function<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    prompt_tokens_details: UsageDetails,
}

#[derive(Serialize)]
struct UsageDetails {
    cached_tokens: usize,
}

/// The read-tool calls a run makes, in order, before it submits.
const SCRIPT: [Function<'static>; 3] = [
    Function {
        name: "list_problem_pods",
        arguments: r#"{"namespace":"default"}"#,
    },
    Function {
        name: "list_resources",
        arguments: r#"{"kind":"pod","namespace":"default"}"#,
    },
    Function {
        name: "get_events",
        arguments: r#"{"namespace":"default","object_name":"","limit":20}"#,
    },
];

/// Ends a run as inconclusive, the one answer that cites nothing.
const SUBMIT: Function<'static> = Function {
    name: "submit_diagnosis",
    arguments: r#"{"status":"inconclusive","findings":[],"summary":"fake worker: no model behind this answer"}"#,
};

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// The answer every successful chat gets. The one assistant message calls
/// the next tool in `SCRIPT` until the request carries `steps` tool results,
/// then `submit_diagnosis`.
fn completion(request: &ChatRequest, body_len: usize, cached: usize, steps: usize) -> Bytes {
    let done = request.tool_results;
    let call = if done < steps.min(SCRIPT.len()) {
        SCRIPT[done]
    } else {
        SUBMIT
    };
    let prompt = tokens(body_len);
    to_go_json(&ChatCompletion {
        id: "chatcmpl-fake",
        object: "chat.completion",
        created: now_unix(),
        model: &request.model,
        choices: [Choice {
            index: 0,
            message: Message {
                role: "assistant",
                content: None,
                tool_calls: Some([ToolCall {
                    id: format!("call_fake_{}", done + 1),
                    r#type: "function",
                    function: call,
                }]),
            },
            finish_reason: "tool_calls",
        }],
        usage: Usage {
            prompt_tokens: prompt,
            completion_tokens: 35,
            total_tokens: prompt + 35,
            prompt_tokens_details: UsageDetails {
                cached_tokens: cached,
            },
        },
    })
}

/// The answer to a `do_remote_decode` request: one token and a length stop,
/// the only finish for which vLLM's MooncakeConnector holds the blocks.
fn held(body: &[u8]) -> Bytes {
    let prompt = tokens(body.len());
    to_go_json(&ChatCompletion {
        id: "chatcmpl-fake-hold",
        object: "chat.completion",
        created: now_unix(),
        model: "",
        choices: [Choice {
            index: 0,
            message: Message {
                role: "assistant",
                content: Some(""),
                tool_calls: None,
            },
            finish_reason: "length",
        }],
        usage: Usage {
            prompt_tokens: prompt,
            completion_tokens: 1,
            total_tokens: prompt + 1,
            prompt_tokens_details: UsageDetails { cached_tokens: 0 },
        },
    })
}

/// The OpenAI error shape vLLM uses for a failed request. The keys are in
/// alphabetical order, the order Go writes a map in.
fn error_body(status: u16) -> Bytes {
    Bytes::from(go_json(&json!({"error": {
        "code": status,
        "message": format!("fake worker answered {status}"),
        "type": "server_error",
    }})))
}

fn to_go_json(value: &impl Serialize) -> Bytes {
    let text = serde_json::to_string(value).unwrap_or_default();
    Bytes::from(go_escape(text))
}

fn go_json(value: &Value) -> String {
    go_escape(value.to_string())
}

/// Go's `json.Marshal` also escapes these, which can reach the output
/// through the echoed model name or engine id.
fn go_escape(text: String) -> String {
    text.replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

/// ceil(n / 3.5), the byte-to-token rule the gateway uses, so the two agree.
fn tokens(n: usize) -> usize {
    (2 * n).div_ceil(7)
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn json_response(status: StatusCode, body: Bytes) -> Response {
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, JSON);
    response
}

/// Answers as Go's `http.Error` does: plain text, nosniff, and a newline
/// after the message.
fn plain(status: StatusCode, message: &str) -> Response {
    let mut response = Response::new(Body::from(format!("{message}\n")));
    *response.status_mut() = status;
    let headers = response.headers_mut();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    response
}

fn method_not_allowed(allow: &'static str) -> Response {
    let mut response = plain(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed");
    response
        .headers_mut()
        .insert(ALLOW, HeaderValue::from_static(allow));
    response
}
