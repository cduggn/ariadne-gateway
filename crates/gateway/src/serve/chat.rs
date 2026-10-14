//! The proxy path: guard, admit, forward, relay.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use ariadne_decide::{
    Headers, Refusal, Request as Call, RunUsage, Timestamp, inspect, may_leave, parse_data_class,
    parse_priority, parse_request_id,
};
use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::Value;
use tokio::time::Instant;

use super::wire::{ApiError, ApiErrorBody, go_text, sniff, usage_of};
use super::{Event, Inner, MAX_RESPONSE_BYTES, Placed, Reason, json, millis_text, response};
use crate::fleet::{Admission, Decision, Outcome, Ticket, read_capped};
use crate::hop::{self, Move, Report};
use crate::overflow::{self, Attempt};

/// The request headers copied to the worker. Everything else, hop-by-hop headers included,
/// stops at the gateway.
const FORWARDED: [HeaderName; 8] = [
    header::CONTENT_TYPE,
    header::AUTHORIZATION,
    header::ACCEPT,
    HeaderName::from_static("x-request-id"),
    HeaderName::from_static("x-tenant"),
    HeaderName::from_static("x-app"),
    HeaderName::from_static("x-priority"),
    HeaderName::from_static("x-data-class"),
];

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// The headers decide reads, as the text Go's `Header.Get` returned.
struct HeaderText {
    request_id: String,
    tenant: String,
    app: String,
    priority: String,
    data_class: String,
}

impl HeaderText {
    fn of(h: &HeaderMap) -> Self {
        let get = |name: &str| {
            h.get(name)
                .map_or_else(String::new, |v| go_text(v.as_bytes()))
        };
        Self {
            request_id: get("x-request-id"),
            tenant: get("x-tenant"),
            app: get("x-app"),
            priority: get("x-priority"),
            data_class: get("x-data-class"),
        }
    }

    fn headers(&self) -> Headers<'_> {
        Headers {
            request_id: &self.request_id,
            tenant: &self.tenant,
            app: &self.app,
            priority: &self.priority,
            data_class: &self.data_class,
        }
    }
}

/// The request's Event while the handler runs. Dropping it finishes the Event: logged and
/// handed to `on_request` exactly once. When axum drops the handler future because the client
/// went away, it drops this too, with no status set, which is what a client that left looks
/// like.
struct Record {
    server: Arc<Inner>,
    start: Timestamp,
    ev: Event,
    waiting: Waiting,
}

/// What the handler is awaiting and since when, so a client that leaves mid-wait is recorded
/// as Go recorded it: the call it cancelled counts as failed, with the time it ran.
#[derive(Clone, Copy)]
enum Waiting {
    Nothing,
    Hop(Timestamp),
    Upstream(Timestamp),
    Overflow(Timestamp),
}

impl Drop for Record {
    fn drop(&mut self) {
        let now = self.server.opt.clock.now();
        self.ev.total = now.saturating_duration_since(self.start);
        if self.ev.status.is_none() {
            self.ev.reason = Some(Reason::ClientGone);
            match self.waiting {
                Waiting::Nothing => {}
                Waiting::Hop(t) => {
                    self.ev.hop = Some(hop::Outcome::Failed);
                    self.ev.hop_time = now.saturating_duration_since(t);
                }
                Waiting::Upstream(t) => self.ev.upstream = now.saturating_duration_since(t),
                Waiting::Overflow(t) => {
                    self.ev.overflow = Some(overflow::Outcome::Failed);
                    self.ev.overflow_time = now.saturating_duration_since(t);
                }
            }
        }
        self.server.finish(&self.ev);
    }
}

/// An Event from the headers alone, so a request the guard rejects is still logged with what
/// the headers said.
fn event_for(h: &HeaderText) -> Event {
    let (run, step) = parse_request_id(&h.request_id).map_or((None, 0), |(r, s)| (Some(r), s));
    Event {
        request_id: h.request_id.clone(),
        run,
        step,
        tenant: String::new(),
        priority: parse_priority(&h.priority),
        class: parse_data_class(&h.data_class),
        placement: None,
        status: None,
        reason: None,
        gateway: Duration::ZERO,
        queue: Duration::ZERO,
        upstream: Duration::ZERO,
        total: Duration::ZERO,
        usage: None,
        overflow: None,
        overflow_time: Duration::ZERO,
        hop: None,
        hop_time: Duration::ZERO,
        hop_fallback: false,
    }
}

pub(super) async fn chat(s: Arc<Inner>, req: Request) -> Response {
    let start = s.opt.clock.now();
    let (parts, body) = req.into_parts();
    let h = HeaderText::of(&parts.headers);
    let mut rec = Record {
        server: Arc::clone(&s),
        start,
        ev: event_for(&h),
        waiting: Waiting::Nothing,
    };

    let Some(body) = read_body(body, s.opt.max_body.saturating_add(1)).await else {
        // The client went away mid-body; nobody reads this.
        return response(StatusCode::OK, Body::empty());
    };

    let call = match inspect(&body, h.headers(), start, s.opt.budgets, s.opt.max_body) {
        Ok(call) => call,
        Err(reject) => {
            rec.ev.status = Some(400);
            rec.ev.reason = Some(Reason::Guard(reject.reason));
            return api_error(
                StatusCode::BAD_REQUEST,
                &reject.detail,
                "invalid_request_error",
                reject.reason.as_str(),
            );
        }
    };
    rec.ev.tenant = call.tenant.clone();

    let admission = s.gate.admit(&call).await;
    let d = admission.decision();
    rec.ev.placement = Some(Placed {
        pod: d.placement.pod().unwrap_or("").to_owned(),
        policy: d.placement.policy,
        sticky: d.placement.sticky,
        unknown: d.placement.unknown(),
        est: d.est,
    });
    rec.ev.queue = d.queued;
    match admission {
        Admission::Refused(d) => {
            let Err(refusal) = d.placement.result else {
                return response(StatusCode::INTERNAL_SERVER_ERROR, Body::empty());
            };
            refused(&s, &call, &refusal, body, &mut rec).await
        }
        Admission::Admitted(d, ticket) => {
            admitted(&s, &call, &d, ticket, &parts.headers, body, &mut rec).await
        }
    }
}

/// Reads at most `cap` bytes of the client's body, `None` when the client went away.
async fn read_body(mut body: Body, cap: usize) -> Option<Bytes> {
    let mut out = Vec::new();
    while out.len() < cap {
        let Some(frame) = body.frame().await else {
            break;
        };
        if let Ok(data) = frame.ok()?.into_data() {
            let take = data.len().min(cap - out.len());
            out.extend_from_slice(&data[..take]);
        }
    }
    Some(Bytes::from(out))
}

/// A shed verdict. A 503 the stay-or-leave rule lets go is offered to the overflow backend first.
async fn refused(
    s: &Inner,
    call: &Call,
    refusal: &Refusal,
    body: Bytes,
    rec: &mut Record,
) -> Response {
    if !refusal.stays()
        && let Some(served) = leave(s, call, refusal, body, rec).await
    {
        return served;
    }
    let ev = &mut rec.ev;
    ev.status = Some(refusal.code());
    ev.reason = Some(Reason::Shed(refusal.reason));
    refusal_response(refusal)
}

/// The stay-or-leave step for a 503. It records what the overflow decision did and, when the
/// backend answered, returns that answer to relay. Without a configured backend the only effect
/// is the no_backend or blocked_invariant label.
async fn leave(
    s: &Inner,
    call: &Call,
    refusal: &Refusal,
    body: Bytes,
    rec: &mut Record,
) -> Option<Response> {
    let ev = &mut rec.ev;
    let Some(permit) = may_leave(call, refusal) else {
        ev.overflow = Some(overflow::Outcome::Blocked);
        return None;
    };
    let Some(ov) = &s.opt.overflow else {
        ev.overflow = Some(overflow::Outcome::NoBackend);
        return None;
    };
    let Some(charge) = s.gate.charge_offbox(call) else {
        ev.overflow = Some(overflow::Outcome::Quota);
        return None;
    };
    let body_bytes = body.len();
    rec.waiting = Waiting::Overflow(s.opt.clock.now());
    let attempt = ov.send(permit, body).await;
    rec.waiting = Waiting::Nothing;
    let ev = &mut rec.ev;
    ev.overflow = Some(attempt.outcome());
    ev.overflow_time = attempt.took();
    match attempt {
        Attempt::Sent {
            content_type,
            body: answer,
            took,
        } => {
            ev.status = Some(200);
            ev.reason = None;
            ev.usage = usage_of(&answer);
            let u = ev.usage.clone().unwrap_or_default();
            charge.settle(Outcome::Answered {
                usage: RunUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    body_bytes,
                },
                latency: took,
            });
            let mut r = response(StatusCode::OK, answer.clone());
            let h = r.headers_mut();
            if let Some(ct) = content_type.or_else(|| sniff(&answer).map(HeaderValue::from_static))
            {
                h.insert(header::CONTENT_TYPE, ct);
            }
            insert(h, "x-overflow", overflow::Outcome::Sent.as_str());
            insert(h, "x-overflow-backend", ov.name());
            insert(h, "x-gateway-reason", refusal.reason.as_str());
            insert(
                h,
                "server-timing",
                &format!("overflow;dur={}", millis_text(took)),
            );
            Some(r)
        }
        Attempt::Failed { error, .. } => {
            s.opt.log.warn(
                "overflow failed, answering the 503",
                &[
                    ("request_id", Value::from(call.id.as_str())),
                    ("backend", Value::from(ov.name())),
                    ("err", Value::from(error.to_string())),
                ],
            );
            None
        }
        Attempt::Unavailable | Attempt::Capped => None,
    }
}

/// Forwards an admitted request, through the KV hop when it applies, and relays the answer.
/// Dropping this future, as axum does when the client goes away, drops the upstream call, which
/// closes its connection so the worker stops, and drops the ticket, which settles as failed.
async fn admitted(
    s: &Inner,
    call: &Call,
    d: &Decision,
    ticket: Ticket,
    headers: &HeaderMap,
    body: Bytes,
    rec: &mut Record,
) -> Response {
    let deadline = Instant::now() + s.opt.upstream_timeout;
    let pod = ticket.pod().to_owned();
    let base = s.urls.get(&pod).map_or("", String::as_str);
    let fwd_headers = forwarded(headers);
    let from = d.from.as_deref().unwrap_or("");

    let marked = match &s.opt.hop {
        Some(hopper) => {
            let m = Move {
                sticky: d.placement.sticky,
                from,
                to: &pod,
                history: d.history,
            };
            rec.waiting = Waiting::Hop(s.opt.clock.now());
            let (out, report) = hopper.move_kv(&m, body.clone()).await;
            rec.waiting = Waiting::Nothing;
            rec.ev.hop = report.outcome();
            rec.ev.hop_time = report.took();
            if let Report::Failed { err, .. } = &report {
                s.opt.log.warn(
                    "kv hop failed, the worker recomputes",
                    &[
                        ("request_id", Value::from(call.id.as_str())),
                        ("from", Value::from(from)),
                        ("to", Value::from(pod.as_str())),
                        ("err", Value::from(err.to_string())),
                    ],
                );
            }
            out
        }
        None => body.clone(),
    };

    rec.waiting = Waiting::Upstream(s.opt.clock.now());
    let up = match (&s.opt.hop, rec.ev.hop) {
        (Some(hopper), Some(hop::Outcome::Hopped)) => {
            let delivery = HopDelivery {
                hopper,
                base,
                from,
                to: &pod,
                call,
            };
            delivery
                .run(s, body.clone(), marked, fwd_headers, deadline, &mut rec.ev)
                .await
        }
        _ => forward(s, base, marked, fwd_headers, deadline).await,
    };
    rec.waiting = Waiting::Nothing;
    rec.ev.upstream = up.took;

    let answer = match up.result {
        Ok(answer) => answer,
        Err(e) => {
            rec.ev.status = Some(502);
            rec.ev.reason = Some(Reason::UpstreamError);
            return api_error(
                StatusCode::BAD_GATEWAY,
                &format!("upstream request failed: {e}"),
                "upstream_error",
                "upstream_error",
            );
        }
    };

    rec.ev.status = Some(answer.status.as_u16());
    let mut outcome = Outcome::Failed;
    if answer.status == StatusCode::OK {
        rec.ev.usage = usage_of(&answer.body);
        if let Some(u) = &rec.ev.usage {
            outcome = Outcome::Answered {
                usage: RunUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    body_bytes: body.len(),
                },
                latency: up.took,
            };
        }
    }
    ticket.settle(outcome);
    rec.ev.gateway = s
        .opt
        .clock
        .now()
        .saturating_duration_since(rec.start)
        .saturating_sub(rec.ev.queue)
        .saturating_sub(rec.ev.hop_time)
        .saturating_sub(rec.ev.upstream);
    relay(answer, &rec.ev)
}

/// A hopped request's delivery. The marked body tells `to` to pull the run's KV from `from`,
/// bounded by the hop's receive timeout rather than the upstream timeout: a destination pulling
/// from an engine that is gone, because the source restarted behind a cached engine id,
/// otherwise hangs until the upstream timeout (114 s on 2026-10-12). When the destination gives
/// no answer or a 5xx, the source's engine id is forgotten, the hop counts as failed, and the
/// unmarked body goes once more under what is left of the upstream timeout, so `to` recomputes
/// the history. The second attempt carries the request id with a -recompute suffix, because
/// vLLM refuses a request id it still holds and the first attempt may not have been aborted
/// yet. A 4xx would repeat without the hop, so it is relayed as it is.
struct HopDelivery<'a> {
    hopper: &'a hop::Hopper,
    base: &'a str,
    from: &'a str,
    to: &'a str,
    call: &'a Call,
}

impl HopDelivery<'_> {
    async fn run(
        &self,
        s: &Inner,
        body: Bytes,
        marked: Bytes,
        headers: HeaderMap,
        deadline: Instant,
        ev: &mut Event,
    ) -> Reply {
        let receive = deadline.min(Instant::now() + self.hopper.receive_timeout());
        let first = forward(s, self.base, marked, headers.clone(), receive).await;
        let status = first.result.as_ref().ok().map(|a| a.status.as_u16());
        let fallback = status.is_none_or(|code| code >= 500);
        if Instant::now() >= deadline || !fallback {
            return first;
        }
        self.hopper.forget(self.from);
        ev.hop = Some(hop::Outcome::Failed);
        ev.hop_fallback = true;
        let err = first
            .result
            .as_ref()
            .err()
            .map_or(Value::Null, |e| Value::from(e.to_string()));
        s.opt.log.warn(
            "hopped request failed, sending it again without the hop",
            &[
                ("request_id", Value::from(self.call.id.as_str())),
                ("from", Value::from(self.from)),
                ("to", Value::from(self.to)),
                ("status", Value::from(status.unwrap_or(0))),
                ("err", err),
            ],
        );
        let mut again = forward(s, self.base, body, recompute(headers), deadline).await;
        again.took += first.took;
        again
    }
}

/// `in`'s allowlisted headers, each taken when non-empty.
fn forwarded(in_: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for name in FORWARDED {
        if let Some(v) = in_.get(&name).filter(|v| !v.is_empty()) {
            out.insert(name, v.clone());
        }
    }
    out
}

/// `headers` with "-recompute" appended to a non-empty `X-Request-Id`.
fn recompute(mut headers: HeaderMap) -> HeaderMap {
    if let Some(id) = headers.get(X_REQUEST_ID) {
        let mut v = id.as_bytes().to_vec();
        v.extend_from_slice(b"-recompute");
        if let Ok(v) = HeaderValue::from_bytes(&v) {
            headers.insert(X_REQUEST_ID, v);
        }
    }
    headers
}

/// One worker's answer, read in full up to the response cap.
struct Answer {
    status: StatusCode,
    content_type: Option<HeaderValue>,
    body: Vec<u8>,
}

/// An upstream call's result with the time it took, which is reported either way.
struct Reply {
    took: Duration,
    result: Result<Answer, ForwardError>,
}

/// Why an upstream call gave no answer, worded as Go's `http.Client` errors were.
#[derive(Debug)]
enum ForwardError {
    Timeout { url: String },
    Transport { url: String, err: reqwest::Error },
}

impl fmt::Display for ForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout { url } => write!(f, "Post {url:?}: context deadline exceeded"),
            Self::Transport { url, err } => {
                let mut cause: &dyn std::error::Error = err;
                while let Some(next) = cause.source() {
                    cause = next;
                }
                let refused = cause
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::ConnectionRefused);
                match (
                    refused,
                    err.url().and_then(|u| u.socket_addrs(|| None).ok()),
                ) {
                    (true, Some(addrs)) if !addrs.is_empty() => write!(
                        f,
                        "Post {url:?}: dial tcp {}: connect: connection refused",
                        addrs[0]
                    ),
                    _ => write!(f, "Post {url:?}: {cause}"),
                }
            }
        }
    }
}

/// Posts `body` to `base`'s chat endpoint and reads the whole answer, all before `deadline`.
async fn forward(
    s: &Inner,
    base: &str,
    body: Bytes,
    headers: HeaderMap,
    deadline: Instant,
) -> Reply {
    let start = s.opt.clock.now();
    let url = format!("{base}/v1/chat/completions");
    let call = async {
        let resp = s
            .opt
            .client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await?;
        let status = resp.status();
        let content_type = resp.headers().get(header::CONTENT_TYPE).cloned();
        let body = read_capped(resp, MAX_RESPONSE_BYTES).await?;
        Ok(Answer {
            status,
            content_type,
            body,
        })
    };
    let result = match tokio::time::timeout_at(deadline, call).await {
        Ok(r) => r.map_err(|err| ForwardError::Transport {
            url: url.clone(),
            err,
        }),
        Err(_) => Err(ForwardError::Timeout { url }),
    };
    Reply {
        took: s.opt.clock.now().saturating_duration_since(start),
        result,
    }
}

/// The worker's status and body unchanged, plus the gateway's placement and timing headers.
fn relay(a: Answer, ev: &Event) -> Response {
    let ct = a
        .content_type
        .or_else(|| sniff(&a.body).map(HeaderValue::from_static));
    let mut r = response(a.status, a.body);
    let h = r.headers_mut();
    if let Some(ct) = ct {
        h.insert(header::CONTENT_TYPE, ct);
    }
    if let Some(p) = &ev.placement {
        insert(h, "x-pod", &p.pod);
        insert(h, "x-policy", p.policy.as_str());
        insert(h, "x-sticky", p.sticky.as_str());
    }
    if let Some(hop) = ev.hop {
        insert(h, "x-hop", hop.as_str());
    }
    insert(h, "x-gateway-queue-ms", &millis_text(ev.queue));
    insert(
        h,
        "server-timing",
        &format!(
            "gateway;dur={}, queue;dur={}, upstream;dur={}",
            millis_text(ev.gateway),
            millis_text(ev.queue),
            millis_text(ev.upstream)
        ),
    );
    r
}

fn insert(h: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        h.insert(HeaderName::from_static(name), v);
    }
}

fn api_error(status: StatusCode, message: &str, kind: &str, code: &str) -> Response {
    json(
        status,
        &ApiError {
            error: ApiErrorBody {
                message,
                kind,
                code,
            },
        },
    )
}

/// A shed verdict's answer. The client reads only the status code, so the reason and the retry
/// hint go in headers as well as the body.
fn refusal_response(v: &Refusal) -> Response {
    let kind = if v.code() == 429 {
        "rate_limit_exceeded"
    } else {
        "overloaded"
    };
    let status = StatusCode::from_u16(v.code()).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
    let mut r = api_error(status, &v.detail, kind, v.reason.as_str());
    let h = r.headers_mut();
    insert(
        h,
        "retry-after",
        &(v.retry_after.as_secs_f64().ceil() as i64).to_string(),
    );
    insert(h, "x-gateway-reason", v.reason.as_str());
    r
}
