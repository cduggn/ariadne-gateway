use std::time::Duration;

use serde_json::Value;

use crate::time::Timestamp;

/// The agent run a request belongs to: the `X-Request-Id` text before the final `-s<n>` suffix.
/// Never empty. A request with no run carries `None` and gets no stickiness.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(String);

impl RunId {
    /// Returns `None` for an empty name, so "no run" has exactly one representation.
    pub fn new(name: impl Into<String>) -> Option<Self> {
        let name = name.into();
        (!name.is_empty()).then_some(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// `X-Priority`. Any value other than "interactive" is `Batch`, so a missing or garbled header
/// can never jump the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Priority {
    #[default]
    Batch,
    Interactive,
}

impl Priority {
    /// The metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Interactive => "interactive",
        }
    }
}

/// `X-Data-Class`. The default is `Restricted`, and parsing maps every value other than
/// "internal" or "public" to it. A missing header, a typo or a new class name therefore keeps
/// the request on the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DataClass {
    #[default]
    Restricted,
    Internal,
    Public,
}

impl DataClass {
    /// The metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Restricted => "restricted",
            Self::Internal => "internal",
            Self::Public => "public",
        }
    }
}

/// Trims spaces and ignores case. An unknown input is `Batch`, the safe lane.
pub fn parse_priority(s: &str) -> Priority {
    if s.trim().eq_ignore_ascii_case("interactive") {
        Priority::Interactive
    } else {
        Priority::Batch
    }
}

/// Trims spaces and ignores case. An unknown or empty input is `Restricted`, the safe class.
pub fn parse_data_class(s: &str) -> DataClass {
    let s = s.trim();
    if s.eq_ignore_ascii_case("internal") {
        DataClass::Internal
    } else if s.eq_ignore_ascii_case("public") {
        DataClass::Public
    } else {
        DataClass::Restricted
    }
}

/// Splits `<run>-s<n>` at the final `-s<digits>` suffix. Returns `None` when the id has no such
/// suffix, an empty run part, or a non-digit tail. The doctor client builds ids the same way.
pub fn parse_request_id(id: &str) -> Option<(RunId, u64)> {
    let (run, tail) = id.rsplit_once("-s")?;
    if !tail.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // Parsed as i64 because the Go gateway did: a step past i64::MAX means the id has no run.
    let step = u64::try_from(tail.parse::<i64>().ok()?).ok()?;
    Some((RunId::new(run)?, step))
}

/// The header values `inspect` reads. The shell copies them out of the HTTP request, so this
/// crate never sees an HTTP type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Headers<'a> {
    pub request_id: &'a str,
    pub tenant: &'a str,
    pub app: &'a str,
    pub priority: &'a str,
    pub data_class: &'a str,
}

/// Output token ceiling for a request that sets neither `max_completion_tokens` nor `max_tokens`.
pub const DEFAULT_MAX_OUT: u64 = 768;

/// The largest doctor context is roughly 100 KiB, so 1 MiB leaves ample headroom.
pub const DEFAULT_MAX_BODY: usize = 1 << 20;

/// The validated, typed view of one inbound call. `inspect` builds it, and later steps trust
/// every field. It holds the body's length, not its bytes, because the proxy forwards the
/// original bytes untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// `X-Request-Id` verbatim, forwarded unchanged.
    pub id: String,
    pub run: Option<RunId>,
    /// `n` from `-s<n>`; 0 when the id has no run.
    pub step: u64,
    /// Sanitised `X-Tenant`, default "platform".
    pub tenant: String,
    /// `X-App`, for labels only.
    pub app: String,
    pub priority: Priority,
    pub class: DataClass,
    pub body_bytes: usize,
    /// `max_completion_tokens`, else `max_tokens`, else `DEFAULT_MAX_OUT`.
    pub max_out: u64,
    pub arrived: Timestamp,
    /// `arrived` plus the per-priority queue budget.
    pub deadline: Timestamp,
}

/// Per-priority queue budgets, used to stamp `Request::deadline`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budgets {
    pub interactive: Duration,
    pub batch: Duration,
}

/// Both budgets sit well under the client's 120s timeout.
pub const DEFAULT_BUDGETS: Budgets = Budgets {
    interactive: Duration::from_secs(10),
    batch: Duration::from_secs(30),
};

/// Why `inspect` refused a body. `as_str` is the error code and the log reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectReason {
    BodyTooLarge,
    BadJson,
    NoMessages,
    NoModel,
    StreamUnsupported,
    KvTransferParams,
    BadMaxTokens,
}

impl RejectReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BodyTooLarge => "body_too_large",
            Self::BadJson => "bad_json",
            Self::NoMessages => "no_messages",
            Self::NoModel => "no_model",
            Self::StreamUnsupported => "stream_unsupported",
            Self::KvTransferParams => "kv_transfer_params",
            Self::BadMaxTokens => "bad_max_tokens",
        }
    }
}

/// `inspect`'s verdict for a malformed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reject {
    pub reason: RejectReason,
    pub detail: String,
}

impl Reject {
    /// The guard rejects bad input and never judges capacity, so every reject is a 400.
    pub const CODE: u16 = 400;
}

fn reject(reason: RejectReason, detail: impl Into<String>) -> Reject {
    Reject {
        reason,
        detail: detail.into(),
    }
}

/// Validates a raw request body and headers. It checks, in order, the body size, that the body
/// is a JSON object, a non-empty `messages` array, a non-empty `model`, that `stream` is not
/// set, that the client did not set `kv_transfer_params`, and the output token limit. Headers
/// never cause a reject.
pub fn inspect(
    body: &[u8],
    h: Headers<'_>,
    now: Timestamp,
    budgets: Budgets,
    max_body: usize,
) -> Result<Request, Reject> {
    use RejectReason::*;

    if body.len() > max_body {
        return Err(reject(
            BodyTooLarge,
            format!("body exceeds {max_body} bytes"),
        ));
    }

    let top: Value =
        serde_json::from_slice(body).map_err(|_| reject(BadJson, "body is not valid JSON"))?;
    let Value::Object(obj) = top else {
        return Err(reject(BadJson, "body must be a JSON object"));
    };

    let has_messages = obj
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|m| !m.is_empty());
    if !has_messages {
        return Err(reject(NoMessages, "messages must be a non-empty array"));
    }

    let has_model = obj
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|m| !m.is_empty());
    if !has_model {
        return Err(reject(NoModel, "model must be a non-empty string"));
    }

    if obj.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(reject(
            StreamUnsupported,
            "stream is not supported; the gateway serves non-streaming only",
        ));
    }

    // kv_transfer_params is the gateway's own instruction to a worker running vLLM's
    // MooncakeConnector. From a client it could make a worker connect to any address, or pull
    // another request's KV under its transfer id, so the guard refuses it whether or not the hop
    // is on.
    if obj.contains_key("kv_transfer_params") {
        return Err(reject(
            KvTransferParams,
            "kv_transfer_params is set by the gateway, never by a client",
        ));
    }

    let limit = ["max_completion_tokens", "max_tokens"]
        .into_iter()
        .find_map(|key| obj.get(key).map(|v| (key, v)));
    let max_out = match limit {
        None => DEFAULT_MAX_OUT,
        Some((key, v)) => positive_int(v)
            .ok_or_else(|| reject(BadMaxTokens, format!("{key} must be a positive integer")))?,
    };

    let (run, step) = match parse_request_id(h.request_id) {
        Some((run, step)) => (Some(run), step),
        None => (None, 0),
    };
    let priority = parse_priority(h.priority);
    let budget = match priority {
        Priority::Interactive => budgets.interactive,
        Priority::Batch => budgets.batch,
    };

    Ok(Request {
        id: h.request_id.to_owned(),
        run,
        step,
        tenant: sanitize_tenant(h.tenant),
        app: h.app.to_owned(),
        priority,
        class: parse_data_class(h.data_class),
        body_bytes: body.len(),
        max_out,
        arrived: now,
        deadline: now.saturating_add(budget),
    })
}

/// A JSON number that is a positive whole number. A value past `u64::MAX` saturates.
fn positive_int(v: &Value) -> Option<u64> {
    let f = v.as_f64()?;
    (f > 0.0 && f == f.trunc()).then_some(f as u64)
}

/// Lowercases `raw`, keeps only `[a-z0-9._-]`, and caps the result at 64 characters. Empty
/// input, or input with nothing left after filtering, becomes "platform".
fn sanitize_tenant(raw: &str) -> String {
    let kept: String = raw
        .trim()
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|c| matches!(c, 'a'..='z' | '0'..='9' | '.' | '_' | '-'))
        .take(64)
        .collect();
    if kept.is_empty() {
        "platform".to_owned()
    } else {
        kept
    }
}
