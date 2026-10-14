//! The gateway's HTTP shell. It reads each request, asks decide and fleet what to do, forwards
//! the body to the chosen worker and relays the answer. It holds no policy of its own. The body
//! goes unchanged unless the optional KV hop marks it for the worker to pull a moved run's cache.

mod chat;
mod event;
mod mux;
mod wire;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ariadne_decide::{Budgets, DEFAULT_BUDGETS, DEFAULT_MAX_BODY, WorkerState};
use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::Response;
use http::{HeaderValue, Method, StatusCode, header};
use serde_json::Value;

pub use event::{Event, Placed, Reason, Usage};

use self::mux::{Dispatch, Route};
use crate::clock::Clock;
use crate::fleet::{Fleet, Gate};
use crate::hop::Hopper;
use crate::log::{self, Flavor, Logger};
use crate::overflow::Overflow;

/// Bounds one upstream call. It sits under the client's 120 s timeout so the client always hears
/// from the gateway.
pub const DEFAULT_UPSTREAM_TIMEOUT: Duration = Duration::from_secs(115);

/// Caps one upstream response body.
const MAX_RESPONSE_BYTES: usize = 32 << 20;

/// Sees one `Event` per request, including refusals, 400s and clients that went away.
pub type OnRequest = Arc<dyn Fn(&Event) + Send + Sync>;

/// Renders the Prometheus text exposition for `GET /metrics`.
pub type Exposition = Arc<dyn Fn() -> String + Send + Sync>;

/// What shapes a `Server`. `Default` is the shipped shape with no metrics, no hop and no
/// overflow backend; a `None` overflow answers every 503, which is the shipped behaviour.
pub struct Options {
    pub max_body: usize,
    pub budgets: Budgets,
    pub upstream_timeout: Duration,
    pub client: reqwest::Client,
    pub log: Logger,
    pub clock: Clock,
    pub on_request: Option<OnRequest>,
    /// Serves `GET /metrics` when set; the route does not exist otherwise.
    pub metrics: Option<Exposition>,
    /// Copies a moved run's KV to its new worker; `None` leaves every moved run to recompute.
    pub hop: Option<Arc<Hopper>>,
    /// Offered every 503 that `may_leave` lets leave the box.
    pub overflow: Option<Arc<Overflow>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_body: DEFAULT_MAX_BODY,
            budgets: DEFAULT_BUDGETS,
            upstream_timeout: DEFAULT_UPSTREAM_TIMEOUT,
            client: reqwest::Client::new(),
            log: Logger::discard(),
            clock: Clock::system(),
            on_request: None,
            metrics: None,
            hop: None,
            overflow: None,
        }
    }
}

/// The gateway's handler set over one Gate and one Fleet. Cloning is cheap.
#[derive(Clone)]
pub struct Server(Arc<Inner>);

struct Inner {
    gate: Gate,
    fleet: Fleet,
    /// Each pod's base URL without a trailing slash.
    urls: HashMap<String, String>,
    opt: Options,
}

impl Server {
    pub fn new(
        gate: Gate,
        fleet: Fleet,
        urls: impl IntoIterator<Item = (String, String)>,
        opt: Options,
    ) -> Self {
        let urls = urls
            .into_iter()
            .map(|(pod, base)| {
                let base = base.strip_suffix('/').unwrap_or(&base).to_owned();
                (pod, base)
            })
            .collect();
        Self(Arc::new(Inner {
            gate,
            fleet,
            urls,
            opt,
        }))
    }

    /// Every route, dispatched the way Go's `http.ServeMux` did.
    pub fn router(&self) -> Router {
        Router::new().fallback(serve).with_state(self.clone())
    }
}

async fn serve(State(s): State<Server>, req: Request) -> Response {
    let uri = req.uri().clone();
    let route = mux::dispatch(
        req.method(),
        uri.path(),
        uri.query(),
        s.0.opt.metrics.is_some(),
    );
    match route {
        Dispatch::Serve(Route::Chat) => chat::chat(Arc::clone(&s.0), req).await,
        Dispatch::Serve(Route::Healthz) => text(StatusCode::OK, "ok"),
        Dispatch::Serve(Route::Readyz) => readyz(&s.0),
        Dispatch::Serve(Route::Workers) => workers(&s.0),
        Dispatch::Serve(Route::WorkerMetrics(pod)) => worker_metrics(&s.0, &pod).await,
        Dispatch::Serve(Route::Metrics) => {
            metrics(&s.0, req.headers().get(header::ACCEPT_ENCODING))
        }
        Dispatch::Redirect(location) => redirect(req.method(), &location),
        Dispatch::MethodNotAllowed { allow } => {
            let mut r = http_error(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed");
            r.headers_mut()
                .insert(header::ALLOW, HeaderValue::from_static(allow));
            r
        }
        Dispatch::NotFound => http_error(StatusCode::NOT_FOUND, "404 page not found"),
        Dispatch::BadRequest => {
            let mut r = response(StatusCode::BAD_REQUEST, "400 Bad Request");
            set(&mut r, header::CONTENT_TYPE, "text/plain; charset=utf-8");
            set(&mut r, header::CONNECTION, "close");
            r
        }
    }
}

/// 200 while any pod is Ready, else 503.
fn readyz(s: &Inner) -> Response {
    if s.gate.view().iter().any(|ws| ws.ready) {
        text(StatusCode::OK, "ready")
    } else {
        http_error(StatusCode::SERVICE_UNAVAILABLE, "no worker is ready")
    }
}

/// Each pod's fleet status next to the gate's view, in the field names and number types Go's
/// `encoding/json` gave `fleet.WorkerStatus` and `decide.WorkerState`.
fn workers(s: &Inner) -> Response {
    let views: HashMap<String, WorkerState> = s
        .gate
        .view()
        .into_iter()
        .map(|ws| (ws.view.pod.clone(), ws))
        .collect();
    let statuses = s.fleet.status();
    let out: Vec<wire::DebugWorker<'_>> = statuses
        .iter()
        .map(|st| wire::DebugWorker::new(st, views.get(&st.pod)))
        .collect();
    json(StatusCode::OK, &out)
}

async fn worker_metrics(s: &Inner, pod: &str) -> Response {
    if !s.urls.contains_key(pod) {
        return http_error(StatusCode::NOT_FOUND, &format!("unknown pod {pod}"));
    }
    match s.fleet.raw_metrics(pod).await {
        Ok(body) => {
            let mut r = response(StatusCode::OK, body);
            set(&mut r, header::CONTENT_TYPE, "text/plain; version=0.0.4");
            set(&mut r, header::X_CONTENT_TYPE_OPTIONS, "nosniff");
            r
        }
        Err(e) => http_error(StatusCode::BAD_GATEWAY, &e.to_string()),
    }
}

/// The exposition as promhttp served it: gzipped when the client accepts gzip.
fn metrics(s: &Inner, accept_encoding: Option<&HeaderValue>) -> Response {
    let text = s
        .opt
        .metrics
        .as_ref()
        .map(|render| render())
        .unwrap_or_default();
    let gzipped = accepts_gzip(accept_encoding)
        .then(|| gzip(text.as_bytes()))
        .flatten();
    let mut r = match gzipped {
        Some(body) => {
            let mut r = response(StatusCode::OK, body);
            set(&mut r, header::CONTENT_ENCODING, "gzip");
            r
        }
        None => response(StatusCode::OK, text),
    };
    set(
        &mut r,
        header::CONTENT_TYPE,
        "text/plain; version=0.0.4; charset=utf-8; escaping=underscores",
    );
    r
}

/// Whether `Accept-Encoding` lists gzip with a non-zero quality.
fn accepts_gzip(v: Option<&HeaderValue>) -> bool {
    let Some(v) = v.and_then(|v| v.to_str().ok()) else {
        return false;
    };
    v.split(',').any(|item| {
        let mut parts = item.split(';').map(str::trim);
        let coding = parts.next().unwrap_or("");
        let q = parts
            .find_map(|p| p.strip_prefix("q="))
            .map_or(Some(1.0), |q| q.parse::<f64>().ok());
        coding.eq_ignore_ascii_case("gzip") && q.is_some_and(|q| q > 0.0)
    })
}

fn gzip(b: &[u8]) -> Option<Vec<u8>> {
    use std::io::Write as _;
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(b).ok()?;
    enc.finish().ok()
}

/// `http.Redirect` with 307, as ServeMux sends for an unclean path.
fn redirect(method: &Method, location: &str) -> Response {
    let get = method == Method::GET;
    let body = if get {
        format!(
            "<a href=\"{}\">Temporary Redirect</a>.\n\n",
            html_escape(location)
        )
    } else {
        String::new()
    };
    let mut r = response(StatusCode::TEMPORARY_REDIRECT, body);
    if let Ok(v) = HeaderValue::try_from(location) {
        r.headers_mut().insert(header::LOCATION, v);
    }
    if get || method == Method::HEAD {
        set(&mut r, header::CONTENT_TYPE, "text/html; charset=utf-8");
    }
    r
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&#34;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

fn response(status: StatusCode, body: impl Into<Body>) -> Response {
    let mut r = Response::new(body.into());
    *r.status_mut() = status;
    r
}

fn set(r: &mut Response, name: header::HeaderName, value: &'static str) {
    r.headers_mut()
        .insert(name, HeaderValue::from_static(value));
}

/// A plain body whose type Go's server sniffed as text.
fn text(status: StatusCode, body: &'static str) -> Response {
    let mut r = response(status, body);
    set(&mut r, header::CONTENT_TYPE, "text/plain; charset=utf-8");
    r
}

/// `http.Error`: the message and a newline as nosniff text.
fn http_error(status: StatusCode, msg: &str) -> Response {
    let mut r = response(status, format!("{msg}\n"));
    set(&mut r, header::CONTENT_TYPE, "text/plain; charset=utf-8");
    set(&mut r, header::X_CONTENT_TYPE_OPTIONS, "nosniff");
    r
}

/// `json.NewEncoder(w).Encode(v)` under `application/json`.
fn json(status: StatusCode, v: &impl serde::Serialize) -> Response {
    let mut body = log::encode(v, Flavor::Json);
    body.push(b'\n');
    let mut r = response(status, body);
    set(&mut r, header::CONTENT_TYPE, "application/json");
    r
}

/// `d` in milliseconds rounded to one decimal, for the log line.
fn millis(d: Duration) -> f64 {
    (d.as_nanos() as f64 / 1e6 * 10.0).round() / 10.0
}

/// `millis` with exactly one decimal, for the headers.
fn millis_text(d: Duration) -> String {
    format!("{:.1}", millis(d))
}

impl Inner {
    /// Writes the request's log line and hands the Event to `on_request`.
    fn finish(&self, ev: &Event) {
        let placed = ev.placement.as_ref();
        let usage = ev.usage.clone().unwrap_or_default();
        self.opt.log.info(
            "request",
            &[
                ("request_id", Value::from(ev.request_id.as_str())),
                (
                    "run",
                    Value::from(ev.run.as_ref().map_or("", |r| r.as_str())),
                ),
                ("step", Value::from(ev.step)),
                ("tenant", Value::from(ev.tenant.as_str())),
                ("priority", Value::from(ev.priority.as_str())),
                ("class", Value::from(ev.class.as_str())),
                ("pod", Value::from(placed.map_or("", |p| p.pod.as_str()))),
                (
                    "policy",
                    Value::from(placed.map_or("", |p| p.policy.as_str())),
                ),
                (
                    "sticky",
                    Value::from(placed.map_or("", |p| p.sticky.as_str())),
                ),
                ("unknown", Value::from(placed.is_some_and(|p| p.unknown))),
                ("est", Value::from(placed.map_or(0, |p| p.est))),
                ("status", Value::from(ev.status.unwrap_or(0))),
                ("reason", Value::from(ev.reason.map_or("", Reason::as_str))),
                ("queue_ms", Value::from(millis(ev.queue))),
                ("upstream_ms", Value::from(millis(ev.upstream))),
                ("total_ms", Value::from(millis(ev.total))),
                ("prompt_tokens", Value::from(usage.prompt_tokens)),
                ("cached_tokens", Value::from(usage.cached_tokens)),
                ("completion_tokens", Value::from(usage.completion_tokens)),
                ("finish_reason", Value::from(usage.finish_reason)),
                (
                    "overflow",
                    Value::from(ev.overflow.map_or("", |o| o.as_str())),
                ),
                ("overflow_ms", Value::from(millis(ev.overflow_time))),
                ("hop", Value::from(ev.hop.map_or("", |h| h.as_str()))),
                ("hop_ms", Value::from(millis(ev.hop_time))),
                ("hop_fallback", Value::from(ev.hop_fallback)),
            ],
        );
        if let Some(f) = &self.opt.on_request {
            f(ev);
        }
    }
}

#[cfg(test)]
mod tests;
