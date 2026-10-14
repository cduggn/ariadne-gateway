//! The Superlinked backend: an OpenAI-compatible chat-completions client with
//! a bearer key.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use ariadne_decide::Offboxable;
use bytes::Bytes;
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderValue, StatusCode};
use reqwest::Url;
use serde_json::value::RawValue;

use super::{Answer, ProbeError, SendError};

/// Shapes the Superlinked backend. The paths are settings because the hosted
/// API's exact routes are unconfirmed (F22 saw only 402s): the defaults are
/// the OpenAI-compatible ones SIE's HTTP reference lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// The backend's origin. A trailing "/" or "/v1" is tolerated and
    /// dropped, because the course lab's OVERFLOW_BASE_URL carried the "/v1".
    pub base_url: String,
    pub chat_path: String,
    /// Fetched by the probe with the key; must answer 2xx.
    pub probe_path: String,
    /// Replaces the request's model name. `None` forwards the client's.
    pub model: Option<String>,
    /// Bounds one chat call. It sits under the client's 120 s.
    pub timeout: Duration,
    /// How often the circuit re-probes. Zero probes only at startup.
    pub probe_interval: Duration,
    pub max_inflight: usize,
}

impl Default for Config {
    /// The hosted Superlinked API as the course lab addressed it.
    fn default() -> Self {
        Self {
            base_url: "https://api.superlinked.com".to_owned(),
            chat_path: "/v1/chat/completions".to_owned(),
            probe_path: "/v1/models".to_owned(),
            model: None,
            timeout: Duration::from_secs(60),
            probe_interval: Duration::from_secs(30),
            max_inflight: 4,
        }
    }
}

/// Why a `Config` cannot be used. The messages are the Go gateway's, which
/// the binary prints at startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    BaseUrl(String),
    RelativePath,
    NonPositiveTimeout,
    NoSlots,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BaseUrl(url) => write!(f, "base url {url:?} is not an http(s) origin"),
            Self::RelativePath => f.write_str("chat path and probe path must start with /"),
            Self::NonPositiveTimeout => f.write_str("timeout must be positive"),
            Self::NoSlots => f.write_str("max inflight must be at least 1"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// `Superlinked::new`'s error: the config error with Go's "overflow config: "
/// prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidConfig(pub ConfigError);

impl fmt::Display for InvalidConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "overflow config: {}", self.0)
    }
}

impl std::error::Error for InvalidConfig {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl Config {
    /// Checks the shape. An error here is a startup error.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !is_http_origin(&self.base_url) {
            return Err(ConfigError::BaseUrl(self.base_url.clone()));
        }
        if !self.chat_path.starts_with('/') || !self.probe_path.starts_with('/') {
            return Err(ConfigError::RelativePath);
        }
        if self.timeout.is_zero() {
            return Err(ConfigError::NonPositiveTimeout);
        }
        if self.max_inflight < 1 {
            return Err(ConfigError::NoSlots);
        }
        Ok(())
    }

    /// `base_url` without a trailing "/" or "/v1".
    fn origin(&self) -> &str {
        let s = self.base_url.strip_suffix('/').unwrap_or(&self.base_url);
        s.strip_suffix("/v1").unwrap_or(s)
    }
}

/// Go's `url.Parse` rejects spaces, control bytes and an empty authority,
/// all of which the WHATWG parser behind `Url` repairs, so those are refused
/// before it sees the text.
fn is_http_origin(base: &str) -> bool {
    let strict = !base.bytes().any(|b| b <= b' ' || b == 0x7f)
        && base
            .split_once("://")
            .is_some_and(|(_, rest)| !rest.is_empty() && !rest.starts_with('/'));
    strict
        && Url::parse(base).is_ok_and(|url| {
            matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some_and(|h| !h.is_empty())
        })
}

/// The backend's bearer key. It never prints: `Debug` shows a placeholder and
/// there is no `Display`.
#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    /// `None` for an empty key, which sends no Authorization header.
    pub fn new(key: String) -> Option<Self> {
        (!key.is_empty()).then_some(Self(key))
    }

    fn bearer(&self) -> Option<HeaderValue> {
        let mut value = HeaderValue::try_from(format!("Bearer {}", self.0)).ok()?;
        value.set_sensitive(true);
        Some(value)
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

/// Caps one backend response body, as serve does for a worker's.
const MAX_RESPONSE_BYTES: usize = 32 << 20;

/// Read from a probe answer so the connection can be reused.
const PROBE_DRAIN_BYTES: usize = 4096;

/// An OpenAI-compatible chat-completions client with a bearer key. It sends
/// the body (with the model rewritten when configured) and nothing else:
/// none of the client's headers, so no run, tenant or request id leaves the
/// box.
#[derive(Debug)]
pub struct Superlinked {
    cfg: Config,
    key: Option<ApiKey>,
    client: reqwest::Client,
}

impl Superlinked {
    /// Without a key the backend is expected to refuse the probe, which
    /// keeps the circuit open.
    pub fn new(
        cfg: Config,
        key: Option<ApiKey>,
        client: reqwest::Client,
    ) -> Result<Self, InvalidConfig> {
        cfg.validate().map_err(InvalidConfig)?;
        Ok(Self { cfg, key, client })
    }

    /// Posts `body` to the chat path. Taking the permit means nothing can
    /// call this for a request the stay-or-leave rule kept on the box.
    pub async fn send(&self, _permit: Offboxable, body: Bytes) -> Result<Answer, SendError> {
        let out = match &self.cfg.model {
            Some(model) => rewrite_model(&body, model)?,
            None => body,
        };
        let url = format!("{}{}", self.cfg.origin(), self.cfg.chat_path);
        let request = self
            .authorize(self.client.post(&url))
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .body(out);
        let call = async {
            let response = request.send().await?;
            let status = response.status();
            let content_type = response.headers().get(CONTENT_TYPE).cloned();
            let body = read_capped(response, MAX_RESPONSE_BYTES).await?;
            Ok(Answer {
                status,
                content_type,
                body: Bytes::from(body),
            })
        };
        tokio::time::timeout(self.cfg.timeout, call)
            .await
            .unwrap_or(Err(SendError::Timeout { url }))
    }

    /// Fetches the probe path with the key. Any 2xx passes. A 401, 402 or 403
    /// means the key or the account is not usable, which is the case F22
    /// found, and reads as unavailable like any other failure.
    pub async fn probe(&self) -> Result<(), ProbeError> {
        let url = format!("{}{}", self.cfg.origin(), self.cfg.probe_path);
        let response = self
            .authorize(self.client.get(&url))
            .header(ACCEPT, "application/json")
            .send()
            .await?;
        let status = response.status();
        let _ = read_capped(response, PROBE_DRAIN_BYTES).await;
        if !status.is_success() {
            return Err(ProbeError::Status {
                path: self.cfg.probe_path.clone(),
                status,
            });
        }
        Ok(())
    }

    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.key.as_ref().and_then(ApiKey::bearer) {
            Some(bearer) => request.header(AUTHORIZATION, bearer),
            None => request,
        }
    }
}

/// Reads at most `cap` bytes of a response and leaves the rest unread.
async fn read_capped(
    mut response: reqwest::Response,
    cap: usize,
) -> Result<Vec<u8>, reqwest::Error> {
    let mut out = Vec::new();
    while out.len() < cap {
        let Some(chunk) = response.chunk().await? else {
            break;
        };
        let room = cap - out.len();
        out.extend_from_slice(&chunk[..chunk.len().min(room)]);
    }
    Ok(out)
}

/// Returns `body` with its top-level "model" set to `model`, written as Go's
/// `json.Marshal` of a `map[string]json.RawMessage` writes it: keys sorted,
/// whitespace dropped, and `<`, `>`, `&`, U+2028 and U+2029 escaped inside
/// strings. The values themselves are otherwise byte for byte.
pub(crate) fn rewrite_model(body: &[u8], model: &str) -> Result<Bytes, SendError> {
    let mut fields: BTreeMap<String, Box<RawValue>> =
        serde_json::from_slice(body).map_err(SendError::Body)?;
    let model = serde_json::value::to_raw_value(model).map_err(SendError::Body)?;
    fields.insert("model".to_owned(), model);
    let out = serde_json::to_vec(&fields).map_err(SendError::Body)?;
    Ok(Bytes::from(go_compact(&out)))
}

/// Go's `json.Compact` with HTML escaping, over JSON already known valid.
fn go_compact(json: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(json.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < json.len() {
        let b = json[i];
        i += 1;
        if !in_string {
            if !matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                out.push(b);
            }
            in_string = b == b'"';
            continue;
        }
        if escaped {
            escaped = false;
            out.push(b);
            continue;
        }
        match b {
            b'\\' => {
                escaped = true;
                out.push(b);
            }
            b'"' => {
                in_string = false;
                out.push(b);
            }
            b'<' | b'>' | b'&' => out.extend_from_slice(format!("\\u{:04x}", b).as_bytes()),
            0xe2 if json.get(i) == Some(&0x80) && matches!(json.get(i + 1), Some(0xa8 | 0xa9)) => {
                let last = json[i + 1];
                out.extend_from_slice(if last == 0xa8 { b"\\u2028" } else { b"\\u2029" });
                i += 2;
            }
            _ => out.push(b),
        }
    }
    out
}

/// Where a failed send's fault lies, which decides whether it opens the
/// circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    /// The backend is unusable for everyone: no answer, an auth or credit
    /// refusal, a rate limit, or a server error.
    Backend,
    /// A 4xx about the one request (400, 413, 422 and the like). One
    /// tenant's bad body must not switch overflow off for every other.
    Request,
}

impl SendError {
    pub(crate) fn fault(&self) -> Fault {
        match self {
            // Go opens the circuit on any error without a status, a body it
            // could not rewrite included. Serve has already parsed the body
            // as a JSON object, so that case does not arise there.
            Self::NoBackend | Self::Body(_) | Self::Transport(_) | Self::Timeout { .. } => {
                Fault::Backend
            }
            Self::Status(status) => match *status {
                StatusCode::UNAUTHORIZED
                | StatusCode::PAYMENT_REQUIRED
                | StatusCode::FORBIDDEN
                | StatusCode::TOO_MANY_REQUESTS => Fault::Backend,
                s if s.is_server_error() || s.as_u16() >= 600 => Fault::Backend,
                _ => Fault::Request,
            },
        }
    }
}
