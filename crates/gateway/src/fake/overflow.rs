//! A fake overflow backend. It serves the two routes the gateway's
//! Superlinked client touches, GET /v1/models as the probe and POST
//! /v1/chat/completions, checks the bearer key when one is configured, and
//! records every chat request it receives, so a test can prove what left the
//! box and what did not. Statuses, headers and bodies match the Go fake's,
//! Go's `net/http` 404 and 405 answers included.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::Response;
use bytes::Bytes;
use http::header::{ALLOW, AUTHORIZATION, CONTENT_TYPE, X_CONTENT_TYPE_OPTIONS};
use http::{HeaderMap, HeaderValue, Method, StatusCode};
use serde_json::{Value, json};

const JSON: HeaderValue = HeaderValue::from_static("application/json");
const MODELS: &str = r#"{"object":"list","data":[{"id":"fake-overflow","object":"model"}]}"#;

/// The knobs a test or a demo turns. `key`, when set, is the bearer key
/// every request must carry, and a request without it gets 401. `down` makes
/// the probe answer 503, as a backend that is out would. A `status` other
/// than 200 makes chat answer that status with an OpenAI error body, which is
/// how a 402 out-of-credit backend looks. `latency` delays every chat answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    pub key: Option<String>,
    pub down: bool,
    pub status: StatusCode,
    pub latency: Duration,
}

/// One chat request as the backend saw it, and the bytes it answered with.
/// `response` is `None` when the request was refused for its key or the
/// client left before the answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recorded {
    pub header: HeaderMap,
    pub body: Bytes,
    pub response: Option<Bytes>,
}

struct Inner {
    settings: Settings,
    requests: Vec<Recorded>,
    probes: usize,
}

/// One fake overflow backend. Clones share the same state, so a test keeps
/// one while the router serves from another.
#[derive(Clone)]
pub struct Backend {
    inner: Arc<Mutex<Inner>>,
}

impl Default for Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend {
    /// Returns a Backend that is up, needs no key and answers 200.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                settings: Settings {
                    key: None,
                    down: false,
                    status: StatusCode::OK,
                    latency: Duration::ZERO,
                },
                requests: Vec::new(),
                probes: 0,
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

    /// Returns a copy of every chat request so far, oldest first.
    pub fn requests(&self) -> Vec<Recorded> {
        self.lock().requests.clone()
    }

    /// How many probe requests the backend has answered.
    pub fn probes(&self) -> usize {
        self.lock().probes
    }

    /// Serves GET /v1/models and POST /v1/chat/completions.
    pub fn router(&self) -> Router {
        // One fallback instead of axum's method routing, whose 404 and 405
        // answers differ from Go's ServeMux in body and Allow header.
        Router::new().fallback(route).with_state(self.clone())
    }

    fn models(&self, headers: &HeaderMap) -> Response {
        let settings = {
            let mut inner = self.lock();
            inner.probes += 1;
            inner.settings.clone()
        };
        if !authorized(headers, &settings) {
            return unauthorized();
        }
        if settings.down {
            return plain(StatusCode::SERVICE_UNAVAILABLE, "backend down");
        }
        json_response(StatusCode::OK, Bytes::from_static(MODELS.as_bytes()))
    }

    /// Records the request, waits `latency`, then answers with one assistant
    /// message that submits an inconclusive diagnosis, so a doctor run that
    /// overflowed still ends. A client that leaves during the wait drops
    /// this future, which leaves the record without a response.
    async fn chat(&self, request: Request) -> Response {
        let (parts, body) = request.into_parts();
        let Ok(body) = axum::body::to_bytes(body, usize::MAX).await else {
            return plain(StatusCode::BAD_REQUEST, "read body");
        };
        let (settings, index) = {
            let mut inner = self.lock();
            inner.requests.push(Recorded {
                header: parts.headers.clone(),
                body: body.clone(),
                response: None,
            });
            (inner.settings.clone(), inner.requests.len() - 1)
        };
        if !authorized(&parts.headers, &settings) {
            return unauthorized();
        }

        tokio::time::sleep(settings.latency).await;

        let response = if settings.status == StatusCode::OK {
            completion(&body)
        } else {
            let code = settings.status.as_u16();
            error_body(code, &format!("fake overflow backend answered {code}"))
        };
        if let Some(record) = self.lock().requests.get_mut(index) {
            record.response = Some(response.clone());
        }
        json_response(settings.status, response)
    }
}

async fn route(State(backend): State<Backend>, request: Request) -> Response {
    let method = request.method();
    match request.uri().path() {
        "/v1/models" if method == Method::GET || method == Method::HEAD => {
            backend.models(request.headers())
        }
        "/v1/models" => method_not_allowed("GET, HEAD"),
        "/v1/chat/completions" if method == Method::POST => backend.chat(request).await,
        "/v1/chat/completions" => method_not_allowed("POST"),
        _ => plain(StatusCode::NOT_FOUND, "404 page not found"),
    }
}

/// Checks the bearer key when one is configured.
fn authorized(headers: &HeaderMap, settings: &Settings) -> bool {
    let Some(key) = &settings.key else {
        return true;
    };
    headers
        .get(AUTHORIZATION)
        .is_some_and(|got| got.as_bytes() == format!("Bearer {key}").as_bytes())
}

fn unauthorized() -> Response {
    json_response(
        StatusCode::UNAUTHORIZED,
        error_body(StatusCode::UNAUTHORIZED.as_u16(), "invalid api key"),
    )
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

/// Serialises as Go's `json.Marshal` does. The `json!` literals in this file
/// list their keys in alphabetical order, which is the order Go writes a map
/// in and so the order serde_json must write whether or not another crate
/// turns on its `preserve_order` feature. Go also escapes the characters
/// below, which can reach the output through the echoed model name.
fn go_json(value: &Value) -> Bytes {
    let text = value
        .to_string()
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    Bytes::from(text)
}

/// The answer every successful chat gets. Its id marks it as the fake's, and
/// its model echoes the request's, so a test can see the gateway's model
/// rewrite. A malformed body reads as an empty model.
fn completion(body: &[u8]) -> Bytes {
    let model = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|request| Some(request.get("model")?.as_str()?.to_owned()))
        .unwrap_or_default();
    let prompt = body.len().div_ceil(4);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    go_json(&json!({
        "choices": [{
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "content": null,
                "role": "assistant",
                "tool_calls": [{
                    "function": {
                        "arguments": r#"{"status":"inconclusive","findings":[],"summary":"fake overflow backend: no model behind this answer"}"#,
                        "name": "submit_diagnosis",
                    },
                    "id": "call_overflow_1",
                    "type": "function",
                }],
            },
        }],
        "created": created,
        "id": "chatcmpl-fake-overflow",
        "model": model,
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 30,
            "prompt_tokens": prompt,
            "total_tokens": prompt + 30,
        },
    }))
}

/// The OpenAI error shape.
fn error_body(status: u16, message: &str) -> Bytes {
    go_json(&json!({"error": {
        "code": status,
        "message": message,
        "type": "server_error",
    }}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{Served, serve};

    const CHAT: &str = r#"{"model":"demo-model","messages":[{"role":"user","content":"hi"}]}"#;

    async fn start() -> (Backend, Served) {
        let backend = Backend::new();
        let served = serve(backend.router()).await.expect("serve the fake");
        (backend, served)
    }

    /// The status, the headers a caller can tell apart, and the body.
    async fn answer(response: reqwest::Response) -> (u16, Vec<(String, String)>, String) {
        let status = response.status().as_u16();
        let mut headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .filter(|(name, _)| !matches!(name.as_str(), "date" | "content-length"))
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
            .collect();
        headers.sort();
        (status, headers, response.text().await.expect("body"))
    }

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn plain_headers() -> Vec<(String, String)> {
        headers(&[
            ("content-type", "text/plain; charset=utf-8"),
            ("x-content-type-options", "nosniff"),
        ])
    }

    #[tokio::test]
    async fn probe_checks_the_key_and_reports_down() {
        let (backend, served) = start().await;
        let client = reqwest::Client::new();
        let models = format!("{}/v1/models", served.url());
        let json = headers(&[("content-type", "application/json")]);

        let open = answer(client.get(&models).send().await.unwrap()).await;
        assert_eq!(open, (200, json.clone(), MODELS.to_owned()));

        backend.set(|s| s.key = Some("k".to_owned()));
        let refused = answer(client.get(&models).send().await.unwrap()).await;
        assert_eq!(
            refused,
            (
                401,
                json.clone(),
                r#"{"error":{"code":401,"message":"invalid api key","type":"server_error"}}"#
                    .to_owned()
            )
        );
        let keyed = answer(client.get(&models).bearer_auth("k").send().await.unwrap()).await;
        assert_eq!(keyed, (200, json, MODELS.to_owned()));

        backend.set(|s| s.down = true);
        let down = answer(client.get(&models).bearer_auth("k").send().await.unwrap()).await;
        assert_eq!(down, (503, plain_headers(), "backend down\n".to_owned()));

        assert_eq!(backend.probes(), 4);
        assert_eq!(backend.requests(), []);
    }

    #[tokio::test]
    async fn chat_records_the_request_and_the_answer() {
        let (backend, served) = start().await;
        let chat = format!("{}/v1/chat/completions", served.url());
        let response = reqwest::Client::new()
            .post(&chat)
            .header("X-Request-Id", "run-s1")
            .body(CHAT)
            .send()
            .await
            .unwrap();
        let (status, got_headers, body) = answer(response).await;

        assert_eq!(status, 200);
        assert_eq!(
            got_headers,
            headers(&[("content-type", "application/json")])
        );
        let created = serde_json::from_str::<Value>(&body).unwrap()["created"]
            .as_u64()
            .expect("created is a unix time");
        assert_eq!(
            body,
            format!(
                r#"{{"choices":[{{"finish_reason":"tool_calls","index":0,"message":{{"content":null,"role":"assistant","tool_calls":[{{"function":{{"arguments":"{{\"status\":\"inconclusive\",\"findings\":[],\"summary\":\"fake overflow backend: no model behind this answer\"}}","name":"submit_diagnosis"}},"id":"call_overflow_1","type":"function"}}]}}}}],"created":{created},"id":"chatcmpl-fake-overflow","model":"demo-model","object":"chat.completion","usage":{{"completion_tokens":30,"prompt_tokens":17,"total_tokens":47}}}}"#
            )
        );

        let recorded = backend.requests();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].body, CHAT);
        assert_eq!(recorded[0].header["x-request-id"], "run-s1");
        assert_eq!(recorded[0].response.as_deref(), Some(body.as_bytes()));
        assert_eq!(backend.probes(), 0);
    }

    #[tokio::test]
    async fn chat_answers_the_configured_status_with_an_error_body() {
        let (backend, served) = start().await;
        backend.set(|s| s.status = StatusCode::PAYMENT_REQUIRED);
        let chat = format!("{}/v1/chat/completions", served.url());
        let got = answer(
            reqwest::Client::new()
                .post(&chat)
                .body(CHAT)
                .send()
                .await
                .unwrap(),
        )
        .await;

        let want = r#"{"error":{"code":402,"message":"fake overflow backend answered 402","type":"server_error"}}"#;
        assert_eq!(
            got,
            (
                402,
                headers(&[("content-type", "application/json")]),
                want.to_owned()
            )
        );
        assert_eq!(
            backend.requests()[0].response.as_deref(),
            Some(want.as_bytes())
        );
    }

    #[tokio::test]
    async fn chat_without_the_key_is_recorded_and_refused() {
        let (backend, served) = start().await;
        backend.set(|s| s.key = Some("k".to_owned()));
        let chat = format!("{}/v1/chat/completions", served.url());
        let response = reqwest::Client::new()
            .post(&chat)
            .bearer_auth("wrong")
            .body(CHAT)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), 401);
        let recorded = backend.requests();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].header["authorization"], "Bearer wrong");
        assert_eq!(recorded[0].response, None);
    }

    #[tokio::test]
    async fn chat_left_during_the_latency_has_no_response() {
        let (backend, served) = start().await;
        backend.set(|s| s.latency = Duration::from_secs(30));
        let chat = format!("{}/v1/chat/completions", served.url());
        let sent = reqwest::Client::new()
            .post(&chat)
            .body(CHAT)
            .timeout(Duration::from_millis(100))
            .send()
            .await;

        assert!(sent.is_err_and(|err| err.is_timeout()));
        let recorded = backend.requests();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].response, None);
    }

    #[tokio::test]
    async fn other_routes_answer_as_go_serve_mux_does() {
        let (backend, served) = start().await;
        let client = reqwest::Client::new();
        let url = |path: &str| format!("{}{path}", served.url());
        let not_found = (404, plain_headers(), "404 page not found\n".to_owned());
        let not_allowed = |allow: &str| {
            let mut headers = plain_headers();
            headers.insert(0, ("allow".to_owned(), allow.to_owned()));
            (405, headers, "Method Not Allowed\n".to_owned())
        };

        let cases = [
            (Method::GET, "/", not_found.clone()),
            (Method::GET, "/v1/models/", not_found.clone()),
            (Method::POST, "/v1/chat", not_found),
            (Method::POST, "/v1/models", not_allowed("GET, HEAD")),
            (Method::GET, "/v1/chat/completions", not_allowed("POST")),
            (Method::PUT, "/v1/chat/completions", not_allowed("POST")),
        ];
        for (method, path, want) in cases {
            let response = client.request(method.clone(), url(path)).send().await;
            assert_eq!(answer(response.unwrap()).await, want, "{method} {path}");
        }

        let head = client.head(url("/v1/models")).send().await.unwrap();
        assert_eq!(head.status(), 200);
        assert_eq!(
            backend.probes(),
            1,
            "HEAD is a probe, as it is for Go's GET pattern"
        );
        assert_eq!(backend.requests(), []);
    }

    #[test]
    fn completion_escapes_the_echoed_model_as_go_does() {
        let body = completion(br#"{"model":"a<b>&c"}"#);
        let text = std::str::from_utf8(&body).unwrap();
        assert!(
            text.contains(r#""model":"a\u003cb\u003e\u0026c","#),
            "{text}"
        );
        let malformed = completion(b"not json");
        let text = std::str::from_utf8(&malformed).unwrap();
        assert!(text.contains(r#""model":"","#), "{text}");
        assert!(
            text.contains(r#""prompt_tokens":2,"total_tokens":32"#),
            "{text}"
        );
    }
}
