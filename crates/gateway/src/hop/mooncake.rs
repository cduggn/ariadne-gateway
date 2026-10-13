//! The KV hop's transport: vLLM's MooncakeConnector protocol.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};

use http::StatusCode;
use http::header::CONTENT_TYPE;
use reqwest::Url;
use serde::Deserialize;
use serde_json::json;
use serde_json::value::{RawValue, to_raw_value};

/// vLLM's VLLM_MOONCAKE_BOOTSTRAP_PORT default. Each worker running
/// MooncakeConnector serves its engine registry there.
pub const DEFAULT_BOOTSTRAP_PORT: u16 = 8998;

/// Caps the source's one-token answer and the registry reply.
const MAX_REPLY_BYTES: usize = 1 << 20;

/// How the gateway reaches one worker. `base_url` is its OpenAI server and
/// `bootstrap_url` its Mooncake bootstrap server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub base_url: String,
    pub bootstrap_url: String,
}

/// A worker URL that has no scheme or host to put the bootstrap port on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapUrlError {
    base: String,
}

impl fmt::Display for BootstrapUrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worker URL {:?} has no scheme or host", self.base)
    }
}

impl std::error::Error for BootstrapUrlError {}

/// Returns `base`'s host with the bootstrap port, the address vLLM's
/// MooncakeConnector listens on beside the OpenAI server.
pub fn bootstrap_url(base: &str, port: u16) -> Result<String, BootstrapUrlError> {
    let url = Url::parse(base).ok();
    match url
        .as_ref()
        .and_then(|url| Some((url.scheme(), url.host_str()?)))
    {
        Some((scheme, host)) if !host.is_empty() => Ok(format!("{scheme}://{host}:{port}")),
        _ => Err(BootstrapUrlError {
            base: base.to_owned(),
        }),
    }
}

/// Tells the destination where to pull a held KV from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Params {
    pub bootstrap_url: String,
    pub engine_id: String,
    pub transfer_id: String,
}

/// Why a hop's transport step failed.
#[derive(Debug)]
pub enum HopError {
    /// The source answered but did not end at its length cap. vLLM's
    /// MooncakeConnector holds the KV only for a request that did, so there
    /// is nothing to pull.
    NotHeld,
    /// The request body is not a JSON object.
    Body(serde_json::Error),
    Transport(reqwest::Error),
    SourceStatus(StatusCode),
    SourceReply(serde_json::Error),
    RegistryStatus(StatusCode),
    RegistryReply(serde_json::Error),
    /// The registry lists no engine for data-parallel rank 0.
    NoEngine,
}

impl fmt::Display for HopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotHeld => f.write_str("source did not hold the KV: finish_reason is not length"),
            Self::Body(err) => write!(f, "request body: {err}"),
            Self::Transport(err) => write!(f, "{err}"),
            Self::SourceStatus(status) => write!(f, "source answered {}", status.as_u16()),
            Self::SourceReply(err) => write!(f, "source reply: {err}"),
            Self::RegistryStatus(status) => {
                write!(f, "bootstrap registry answered {}", status.as_u16())
            }
            Self::RegistryReply(err) => write!(f, "bootstrap registry: {err}"),
            Self::NoEngine => f.write_str("bootstrap registry lists no engine for rank 0"),
        }
    }
}

impl std::error::Error for HopError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Body(err) | Self::SourceReply(err) | Self::RegistryReply(err) => Some(err),
            Self::Transport(err) => Some(err),
            Self::NotHeld | Self::SourceStatus(_) | Self::RegistryStatus(_) | Self::NoEngine => {
                None
            }
        }
    }
}

impl From<reqwest::Error> for HopError {
    fn from(err: reqwest::Error) -> Self {
        Self::Transport(err)
    }
}

/// Speaks vLLM's MooncakeConnector protocol, as vLLM v0.29.0's
/// examples/disaggregated/mooncake_connector proxy does. The source gets the
/// request with do_remote_decode and a one-token cap, prefills it (mostly
/// from its prefix cache) and holds the blocks. The destination gets the
/// request with do_remote_prefill and pulls the blocks it does not already
/// hold. Both workers run the connector with kv_role kv_both.
#[derive(Debug)]
pub struct Mooncake {
    client: reqwest::Client,
    /// Bootstrap URL to the engine id of data-parallel rank 0.
    engines: Mutex<HashMap<String, String>>,
}

impl Mooncake {
    /// Returns a Mooncake that sends with `client`.
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            client,
            engines: Mutex::new(HashMap::new()),
        }
    }

    fn engines(&self) -> MutexGuard<'_, HashMap<String, String>> {
        self.engines.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Makes `src` prefill `body` and hold its KV under `transfer_id`, and
    /// returns the Params the destination needs. On any error it drops the
    /// cached engine id, because the likeliest cause is a restarted worker
    /// with a new one. `forget` drops it for every other sign of a restart.
    ///
    /// Dropping the returned future cancels the send, as cancelling the Go
    /// context did, and also drops the cached engine id.
    pub async fn send(
        &self,
        src: &Endpoint,
        body: &[u8],
        request_id: &str,
        transfer_id: &str,
    ) -> Result<Params, HopError> {
        let engine_id = self.engine(&src.bootstrap_url).await?;
        let mut forget = ForgetUnlessHeld {
            mooncake: self,
            bootstrap: &src.bootstrap_url,
            held: false,
        };
        self.hold(&src.base_url, body, request_id, transfer_id)
            .await?;
        forget.held = true;
        Ok(Params {
            bootstrap_url: src.bootstrap_url.clone(),
            engine_id,
            transfer_id: transfer_id.to_owned(),
        })
    }

    /// Posts the source request and checks that the source kept the blocks.
    async fn hold(
        &self,
        base: &str,
        body: &[u8],
        request_id: &str,
        transfer_id: &str,
    ) -> Result<(), HopError> {
        let out = source_body(body, transfer_id)?;
        let response = self
            .client
            .post(format!("{base}/v1/chat/completions"))
            .header(CONTENT_TYPE, "application/json")
            .header("X-Request-Id", request_id)
            .body(out)
            .send()
            .await?;
        let status = response.status();
        let reply = read_capped(response).await?;
        if status != StatusCode::OK {
            return Err(HopError::SourceStatus(status));
        }

        let reply: SourceReply = serde_json::from_slice(&reply).map_err(HopError::SourceReply)?;
        match reply.choices.first() {
            Some(choice) if choice.finish_reason.as_deref() == Some("length") => Ok(()),
            _ => Err(HopError::NotHeld),
        }
    }

    /// Returns the engine id behind `bootstrap`, asking its registry once
    /// and caching the answer.
    async fn engine(&self, bootstrap: &str) -> Result<String, HopError> {
        if let Some(id) = self.engines().get(bootstrap) {
            return Ok(id.clone());
        }

        let response = self.client.get(format!("{bootstrap}/query")).send().await?;
        if response.status() != StatusCode::OK {
            return Err(HopError::RegistryStatus(response.status()));
        }
        let reply = read_capped(response).await?;
        // Reads one JSON value and ignores anything after it, as Go's
        // json.Decoder does.
        let mut stream = serde_json::Deserializer::from_slice(&reply);
        let mut ranks: HashMap<String, Option<Rank>> =
            Deserialize::deserialize(&mut stream).map_err(HopError::RegistryReply)?;
        let id = ranks
            .remove("0")
            .flatten()
            .and_then(|rank| rank.engine_id)
            .filter(|id| !id.is_empty())
            .ok_or(HopError::NoEngine)?;
        self.engines().insert(bootstrap.to_owned(), id.clone());
        Ok(id)
    }

    /// Drops the cached engine id behind `bootstrap`, so the next send asks
    /// the registry again. A worker that left Ready or restarted has a new
    /// id, and a hold against the old one succeeds while the pull hangs on
    /// an engine that is gone. An unknown or uncached bootstrap is a no-op.
    pub fn forget(&self, bootstrap: &str) {
        self.engines().remove(bootstrap);
    }
}

/// The part of the source's chat answer that says whether it held the KV.
#[derive(Deserialize)]
struct SourceReply {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    finish_reason: Option<String>,
}

/// One data-parallel rank in the bootstrap registry's reply.
#[derive(Deserialize)]
struct Rank {
    #[serde(default)]
    engine_id: Option<String>,
}

/// Forgets the source's engine id when a hold ends any way but held: an
/// error, or the caller's timeout dropping the future mid-request.
struct ForgetUnlessHeld<'a> {
    mooncake: &'a Mooncake,
    bootstrap: &'a str,
    held: bool,
}

impl Drop for ForgetUnlessHeld<'_> {
    fn drop(&mut self) {
        if !self.held {
            self.mooncake.forget(self.bootstrap);
        }
    }
}

/// Reads at most `MAX_REPLY_BYTES` of a reply and leaves the rest unread.
async fn read_capped(mut response: reqwest::Response) -> Result<Vec<u8>, reqwest::Error> {
    let mut reply = Vec::new();
    while reply.len() < MAX_REPLY_BYTES {
        let Some(chunk) = response.chunk().await? else {
            break;
        };
        let room = MAX_REPLY_BYTES - reply.len();
        reply.extend_from_slice(&chunk[..chunk.len().min(room)]);
    }
    Ok(reply)
}

/// A request body's top level. Every value stays as the bytes the client
/// sent, so a rewrite cannot change a field it does not name.
type Fields = BTreeMap<String, Box<RawValue>>;

fn object(body: &[u8]) -> Result<Fields, HopError> {
    serde_json::from_slice(body).map_err(HopError::Body)
}

fn set(fields: &mut Fields, key: &str, value: &serde_json::Value) -> Result<(), HopError> {
    fields.insert(key.to_owned(), to_raw_value(value).map_err(HopError::Body)?);
    Ok(())
}

/// Returns `body` capped to one token, not streamed, and marked for the
/// source to hold its KV.
fn source_body(body: &[u8], transfer_id: &str) -> Result<Vec<u8>, HopError> {
    let mut fields = object(body)?;
    set(
        &mut fields,
        "kv_transfer_params",
        &json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "transfer_id": transfer_id,
        }),
    )?;
    set(&mut fields, "stream", &json!(false))?;
    set(&mut fields, "max_tokens", &json!(1))?;
    if fields.contains_key("max_completion_tokens") {
        set(&mut fields, "max_completion_tokens", &json!(1))?;
    }
    fields.remove("stream_options");
    serde_json::to_vec(&fields).map_err(HopError::Body)
}

/// Returns `body` marked for the destination to pull the KV `params` names.
/// Every other field is kept as it was.
pub fn receive(body: &[u8], params: &Params) -> Result<Vec<u8>, HopError> {
    let mut fields = object(body)?;
    set(
        &mut fields,
        "kv_transfer_params",
        &json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "remote_bootstrap_addr": params.bootstrap_url,
            "remote_engine_id": params.engine_id,
            "transfer_id": params.transfer_id,
        }),
    )?;
    serde_json::to_vec(&fields).map_err(HopError::Body)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::extract::State;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use bytes::Bytes;
    use http::HeaderMap;
    use serde_json::Value;

    use super::*;
    use crate::fake::{Served, serve};

    /// A request shaped like a doctor step that streams nothing and sets
    /// max_completion_tokens, so the rewrite has both caps to handle.
    const DOCTOR_STEP: &str = r#"{"model":"Qwen/Qwen3-8B-AWQ","messages":[{"role":"user","content":"why does checkout-api restart?"}],"tools":[{"type":"function","function":{"name":"submit_diagnosis","parameters":{"type":"object"}}}],"max_completion_tokens":768,"stream_options":{"include_usage":true},"temperature":0.7}"#;

    struct SourceState {
        engine_id: String,
        finish: String,
        status: StatusCode,
        delay: Duration,
        queries: usize,
        bodies: Vec<Bytes>,
        ids: Vec<String>,
    }

    /// A fake vLLM worker running MooncakeConnector: an OpenAI server and a
    /// bootstrap registry on one listener. `finish` is the finish_reason its
    /// chat answer carries.
    struct Source {
        state: Arc<Mutex<SourceState>>,
        served: Served,
    }

    type Shared = State<Arc<Mutex<SourceState>>>;

    async fn query(State(state): Shared) -> Json<Value> {
        let mut s = state.lock().unwrap();
        s.queries += 1;
        Json(json!({"0": {"engine_id": s.engine_id, "dp_rank": 0}}))
    }

    async fn chat(
        State(state): Shared,
        headers: HeaderMap,
        body: Bytes,
    ) -> (StatusCode, Json<Value>) {
        let (status, finish, delay) = {
            let mut s = state.lock().unwrap();
            s.bodies.push(body);
            let id = headers.get("X-Request-Id").and_then(|v| v.to_str().ok());
            s.ids.push(id.unwrap_or_default().to_owned());
            (s.status, s.finish.clone(), s.delay)
        };
        tokio::time::sleep(delay).await;
        (
            status,
            Json(json!({"choices": [{"finish_reason": finish}]})),
        )
    }

    impl Source {
        async fn start() -> Self {
            let state = Arc::new(Mutex::new(SourceState {
                engine_id: "engine-vllm-0".to_owned(),
                finish: "length".to_owned(),
                status: StatusCode::OK,
                delay: Duration::ZERO,
                queries: 0,
                bodies: Vec::new(),
                ids: Vec::new(),
            }));
            let router = Router::new()
                .route("/query", get(query))
                .route("/v1/chat/completions", post(chat))
                .with_state(state.clone());
            let served = serve(router).await.expect("serve the fake source");
            Self { state, served }
        }

        fn set(&self, edit: impl FnOnce(&mut SourceState)) {
            edit(&mut self.state.lock().unwrap());
        }

        /// How many registry queries and chat requests the source got, and
        /// the request ids of the chat requests.
        fn seen(&self) -> (usize, usize, Vec<String>) {
            let s = self.state.lock().unwrap();
            (s.queries, s.bodies.len(), s.ids.clone())
        }

        fn endpoint(&self) -> Endpoint {
            Endpoint {
                base_url: self.served.url(),
                bootstrap_url: self.served.url(),
            }
        }

        fn last_body(&self) -> Value {
            let s = self.state.lock().unwrap();
            let body = s.bodies.last().expect("source received no chat request");
            serde_json::from_slice(body).expect("source body is not JSON")
        }
    }

    fn decode(body: &[u8]) -> Value {
        serde_json::from_slice(body).expect("not JSON")
    }

    fn mooncake() -> Mooncake {
        Mooncake::new(reqwest::Client::new())
    }

    #[tokio::test]
    async fn send_caps_the_source_request_and_marks_it_to_hold() {
        let src = Source::start().await;
        let params = mooncake()
            .send(
                &src.endpoint(),
                DOCTOR_STEP.as_bytes(),
                "run-s2-hop",
                "xfer-run-s2",
            )
            .await
            .expect("send");
        assert_eq!(
            params,
            Params {
                bootstrap_url: src.served.url(),
                engine_id: "engine-vllm-0".to_owned(),
                transfer_id: "xfer-run-s2".to_owned(),
            }
        );

        let got = src.last_body();
        assert_eq!(
            got["kv_transfer_params"],
            json!({"do_remote_decode": true, "do_remote_prefill": false, "transfer_id": "xfer-run-s2"})
        );
        assert_eq!(got["max_tokens"], json!(1));
        assert_eq!(got["max_completion_tokens"], json!(1));
        assert_eq!(got["stream"], json!(false));
        assert_eq!(
            got.get("stream_options"),
            None,
            "stream_options was kept on a non-streamed request"
        );
        let original = decode(DOCTOR_STEP.as_bytes());
        for key in ["model", "messages", "tools", "temperature"] {
            assert_eq!(got[key], original[key], "{key} changed");
        }
        assert_eq!(
            got.as_object().map(|fields| fields.len()),
            Some(8),
            "the rewrite added a field"
        );
        assert_eq!(src.seen().2, ["run-s2-hop"]);
    }

    #[tokio::test]
    async fn send_asks_the_registry_once_and_again_after_a_failure() {
        let src = Source::start().await;
        let m = mooncake();
        let body = DOCTOR_STEP.as_bytes();
        for i in 1..=2 {
            let sent = m.send(&src.endpoint(), body, "id", "x").await;
            assert!(sent.is_ok(), "send #{i} error = {sent:?}");
        }
        assert_eq!(src.seen().0, 1, "registry queries for two sends");

        src.set(|s| s.status = StatusCode::SERVICE_UNAVAILABLE);
        let failed = m.send(&src.endpoint(), body, "id", "x").await;
        assert!(
            matches!(
                &failed,
                Err(HopError::SourceStatus(StatusCode::SERVICE_UNAVAILABLE))
            ),
            "send to a 503 source = {failed:?}"
        );
        src.set(|s| {
            s.status = StatusCode::OK;
            s.engine_id = "engine-vllm-0-restarted".to_owned();
        });
        let params = m
            .send(&src.endpoint(), body, "id", "x")
            .await
            .expect("send after recovery");
        assert_eq!(params.engine_id, "engine-vllm-0-restarted");
        assert_eq!(
            src.seen().0,
            2,
            "want a second registry query after a failure"
        );
    }

    #[tokio::test]
    async fn send_dropped_mid_hold_asks_the_registry_again() {
        let src = Source::start().await;
        let m = mooncake();
        let body = DOCTOR_STEP.as_bytes();
        m.send(&src.endpoint(), body, "id", "x")
            .await
            .expect("send");

        src.set(|s| s.delay = Duration::from_secs(30));
        let endpoint = src.endpoint();
        let timed_out = tokio::time::timeout(
            Duration::from_millis(100),
            m.send(&endpoint, body, "id", "x"),
        )
        .await;
        assert!(timed_out.is_err(), "send outlived its timeout");

        src.set(|s| s.delay = Duration::ZERO);
        m.send(&src.endpoint(), body, "id", "x")
            .await
            .expect("send after the timeout");
        assert_eq!(
            src.seen().0,
            2,
            "want a second registry query after a dropped send"
        );
    }

    #[tokio::test]
    async fn send_fails_when_the_source_did_not_stop_at_its_cap() {
        let src = Source::start().await;
        src.set(|s| s.finish = "stop".to_owned());
        let sent = mooncake()
            .send(&src.endpoint(), DOCTOR_STEP.as_bytes(), "id", "x")
            .await;
        assert!(matches!(sent, Err(HopError::NotHeld)), "send = {sent:?}");
    }

    #[tokio::test]
    async fn send_rejects_a_body_that_is_not_an_object() {
        let src = Source::start().await;
        let sent = mooncake().send(&src.endpoint(), b"[1,2]", "id", "x").await;
        assert!(matches!(sent, Err(HopError::Body(_))), "send = {sent:?}");
        assert_eq!(
            src.seen().1,
            0,
            "a body that is not an object still reached the source"
        );
    }

    #[test]
    fn receive_marks_the_pull_and_keeps_every_other_field() {
        let params = Params {
            bootstrap_url: "http://vllm-0:8998".to_owned(),
            engine_id: "e0".to_owned(),
            transfer_id: "xfer-1".to_owned(),
        };
        let out = receive(DOCTOR_STEP.as_bytes(), &params).expect("receive");
        let mut got = decode(&out);
        assert_eq!(
            got["kv_transfer_params"],
            json!({
                "do_remote_decode": false, "do_remote_prefill": true,
                "remote_bootstrap_addr": "http://vllm-0:8998", "remote_engine_id": "e0", "transfer_id": "xfer-1",
            })
        );
        got.as_object_mut()
            .expect("an object")
            .remove("kv_transfer_params");
        assert_eq!(
            got,
            decode(DOCTOR_STEP.as_bytes()),
            "receive changed other fields"
        );
    }

    #[test]
    fn rewrites_keep_unnamed_values_byte_for_byte() {
        let body = br#"{"seed":18446744073709551616,"temperature":7e-1,"nested":{"a":1,"a":2},"text":"caf\u00e9 <b>"}"#;
        let params = Params {
            bootstrap_url: "http://vllm-0:8998".to_owned(),
            engine_id: "e0".to_owned(),
            transfer_id: "xfer-1".to_owned(),
        };
        assert_eq!(
            String::from_utf8(receive(body, &params).expect("receive")).expect("utf-8"),
            r#"{"kv_transfer_params":{"do_remote_decode":false,"do_remote_prefill":true,"remote_bootstrap_addr":"http://vllm-0:8998","remote_engine_id":"e0","transfer_id":"xfer-1"},"nested":{"a":1,"a":2},"seed":18446744073709551616,"temperature":7e-1,"text":"caf\u00e9 <b>"}"#
        );
        assert_eq!(
            String::from_utf8(source_body(body, "xfer-1").expect("source_body")).expect("utf-8"),
            r#"{"kv_transfer_params":{"do_remote_decode":true,"do_remote_prefill":false,"transfer_id":"xfer-1"},"max_tokens":1,"nested":{"a":1,"a":2},"seed":18446744073709551616,"stream":false,"temperature":7e-1,"text":"caf\u00e9 <b>"}"#
        );
    }

    #[test]
    fn bootstrap_url_swaps_the_port() {
        let cases = [
            (
                "http://vllm-0.vllm.default.svc.cluster.local:8000",
                Some("http://vllm-0.vllm.default.svc.cluster.local:8998"),
            ),
            ("http://127.0.0.1:18001/", Some("http://127.0.0.1:8998")),
            ("vllm-0:8000", None),
        ];
        for (base, want) in cases {
            let got = bootstrap_url(base, DEFAULT_BOOTSTRAP_PORT);
            assert_eq!(
                got.as_deref().ok(),
                want,
                "bootstrap_url({base:?}) = {got:?}"
            );
        }
    }
}
