//! Joins the cost rule and the transport over the gateway's fixed set of
//! workers.

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::sync::Arc;
use std::time::Duration;

use ariadne_decide::Sticky;
use bytes::Bytes;
use rand::TryRng;
use rand::rngs::{SysError, SysRng};
use tokio::sync::Semaphore;

use super::mooncake::{Endpoint, HopError, Mooncake, receive};
use super::policy::{Config, ConfigError, Estimate, Outcome, Plan, Skip, moved};
use crate::goduration::GoDuration;

/// A monotonic reading: the time elapsed since an arbitrary epoch.
pub type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

/// One placement as the hop sees it. `from` is the worker the run was bound
/// to before this request and `to` the worker it was placed on. `history` is
/// the prompt length the run's last settled step reported, which is what
/// `from` holds in its cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move<'a> {
    pub sticky: Sticky,
    pub from: &'a str,
    pub to: &'a str,
    pub history: u64,
}

/// What `Hopper::move_kv` did.
#[derive(Debug)]
pub enum Report {
    /// The move was not one a hop applies to, and nothing should be counted.
    NotApplicable,
    /// The cost rule chose to recompute.
    Skipped(Skip, Estimate),
    /// `max_inflight` hops were already running.
    Busy(Estimate),
    Hopped {
        estimate: Estimate,
        took: Duration,
    },
    Failed {
        estimate: Estimate,
        took: Duration,
        err: MoveError,
    },
}

impl Report {
    /// The metric label and `X-Hop` value, `None` when nothing is counted.
    pub fn outcome(&self) -> Option<Outcome> {
        match self {
            Self::NotApplicable => None,
            Self::Skipped(skip, _) => Some(skip.outcome()),
            Self::Busy(_) => Some(Outcome::Busy),
            Self::Hopped { .. } => Some(Outcome::Hopped),
            Self::Failed { .. } => Some(Outcome::Failed),
        }
    }

    /// The time the hop spent talking to the source, zero unless a hop was
    /// attempted.
    pub fn took(&self) -> Duration {
        match self {
            Self::Hopped { took, .. } | Self::Failed { took, .. } => *took,
            Self::NotApplicable | Self::Skipped(..) | Self::Busy(_) => Duration::ZERO,
        }
    }

    pub fn error(&self) -> Option<&MoveError> {
        match self {
            Self::Failed { err, .. } => Some(err),
            Self::NotApplicable | Self::Skipped(..) | Self::Busy(_) | Self::Hopped { .. } => None,
        }
    }

    pub fn estimate(&self) -> Option<&Estimate> {
        match self {
            Self::NotApplicable => None,
            Self::Skipped(_, estimate)
            | Self::Busy(estimate)
            | Self::Hopped { estimate, .. }
            | Self::Failed { estimate, .. } => Some(estimate),
        }
    }
}

/// Why an attempted hop failed.
#[derive(Debug)]
pub enum MoveError {
    TransferId(SysError),
    /// The source did not hold the KV within `Config::timeout`.
    Timeout(GoDuration),
    Hop(HopError),
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TransferId(err) => write!(f, "transfer id: {err}"),
            Self::Timeout(timeout) => write!(f, "source did not hold the KV within {timeout}"),
            Self::Hop(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for MoveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TransferId(err) => Some(err),
            Self::Hop(err) => Some(err),
            Self::Timeout(_) => None,
        }
    }
}

/// Why `Hopper::new` refused its arguments.
#[derive(Debug)]
pub enum NewError {
    Config(ConfigError),
    NoEndpoints,
}

impl fmt::Display for NewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(err) => write!(f, "hop config: {err}"),
            Self::NoEndpoints => f.write_str("hop needs at least one worker endpoint"),
        }
    }
}

impl std::error::Error for NewError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(err) => Some(err),
            Self::NoEndpoints => None,
        }
    }
}

/// Copies a moved run's KV to its new worker. It is off unless configured,
/// and every failure falls back to recompute, so a hop can make a request
/// faster but never makes it fail.
pub struct Hopper {
    cfg: Config,
    endpoints: HashMap<String, Endpoint>,
    mooncake: Mooncake,
    now: Clock,
    /// One permit per hop in flight, `max_inflight` deep.
    slots: Semaphore,
}

impl Hopper {
    /// Checks `cfg` and returns a Hopper over `endpoints`, which must name
    /// every worker the gateway places on.
    pub fn new(
        cfg: Config,
        endpoints: HashMap<String, Endpoint>,
        client: reqwest::Client,
        now: Clock,
    ) -> Result<Self, NewError> {
        cfg.validate().map_err(NewError::Config)?;
        if endpoints.is_empty() {
            return Err(NewError::NoEndpoints);
        }
        let slots = usize::try_from(cfg.max_inflight)
            .unwrap_or(usize::MAX)
            .min(Semaphore::MAX_PERMITS);
        Ok(Self {
            cfg,
            endpoints,
            mooncake: Mooncake::new(client),
            now,
            slots: Semaphore::new(slots),
        })
    }

    /// Returns the body to forward to `m.to`. When the move qualifies and the
    /// cost rule says hop, it makes `m.from` hold the run's KV and returns
    /// `body` marked for `m.to` to pull it. In every other case, including a
    /// failed hop, it returns `body` unchanged and `m.to` recomputes.
    ///
    /// Dropping the future cancels the hop and frees its slot.
    ///
    /// A hop that succeeds pins the blocks on `m.from` until `m.to` pulls
    /// them. If the destination request never reaches `m.to`, vLLM frees them
    /// after VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT, 480 s by default, so a burst
    /// of abandoned hops costs the source KV for that long.
    pub async fn move_kv(&self, m: &Move<'_>, body: Bytes) -> (Bytes, Report) {
        let src = match self.endpoints.get(m.from) {
            Some(src) if moved(m.sticky) && m.from != m.to => src,
            _ => return (body, Report::NotApplicable),
        };
        let estimate = match self.cfg.decide(m.history) {
            Plan::Hop(estimate) => estimate,
            Plan::Skip(skip, estimate) => return (body, Report::Skipped(skip, estimate)),
        };
        let Ok(_slot) = self.slots.try_acquire() else {
            return (body, Report::Busy(estimate));
        };

        let start = (self.now)();
        let transfer_id = match new_transfer_id() {
            Ok(id) => id,
            Err(err) => {
                let err = MoveError::TransferId(err);
                let took = Duration::ZERO;
                return (
                    body,
                    Report::Failed {
                        estimate,
                        took,
                        err,
                    },
                );
            }
        };
        let sent = tokio::time::timeout(
            self.cfg.timeout.to_std(),
            self.mooncake.send(src, &body, &transfer_id, &transfer_id),
        )
        .await;
        let marked = match sent {
            Err(_elapsed) => Err(MoveError::Timeout(self.cfg.timeout)),
            Ok(sent) => sent
                .and_then(|params| receive(&body, &params))
                .map_err(MoveError::Hop),
        };
        let took = (self.now)().saturating_sub(start);
        match marked {
            Ok(marked) => (Bytes::from(marked), Report::Hopped { estimate, took }),
            Err(err) => (
                body,
                Report::Failed {
                    estimate,
                    took,
                    err,
                },
            ),
        }
    }

    /// Drops what the Hopper cached about `pod`'s engine, so the next hop off
    /// `pod` asks its bootstrap registry again. The fleet calls it when `pod`
    /// leaves Ready or restarts, and serve when a destination fails to pull
    /// from `pod`. An unknown pod is a no-op.
    pub fn forget(&self, pod: &str) {
        if let Some(src) = self.endpoints.get(pod) {
            self.mooncake.forget(&src.bootstrap_url);
        }
    }

    /// How long the destination request of a hopped run may take before the
    /// gateway sends it again without the hop.
    pub fn receive_timeout(&self) -> Duration {
        self.cfg.receive_timeout.to_std()
    }
}

/// 128 bits from the OS generator. The id names the held blocks on the
/// source, and any request that presents it pulls them, so it is never
/// derived from the client's request id: one tenant cannot name, or collide
/// with, a transfer made for another.
fn new_transfer_id() -> Result<String, SysError> {
    let mut bytes = [0u8; 16];
    SysRng.try_fill_bytes(&mut bytes)?;
    let mut id = String::from("xfer-");
    for b in bytes {
        let _ = write!(id, "{b:02x}");
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::Router;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::{get, post};
    use serde_json::{Value, json};
    use tokio::sync::{mpsc, watch};

    use super::*;
    use crate::fake::{Served, serve};

    const DOCTOR_STEP: &str = r#"{"model":"Qwen/Qwen3-8B-AWQ","messages":[{"role":"user","content":"why does checkout-api restart?"}],"tools":[{"type":"function","function":{"name":"submit_diagnosis","parameters":{"type":"object"}}}],"max_completion_tokens":768,"stream_options":{"include_usage":true},"temperature":0.7}"#;

    /// Hops any history over 10 tokens beyond a 0-token shared prefix.
    const FAST_RULE: Config = Config {
        min_tokens: 10,
        shared_prefix_tokens: 0,
        kv_bytes_per_token: 1,
        transfer_bytes_per_s: 1e12,
        prefill_tokens_per_s: 1.0,
        overhead: GoDuration::from_nanos(0),
        timeout: GoDuration::from_nanos(1_000_000_000),
        receive_timeout: GoDuration::from_nanos(1_000_000_000),
        max_inflight: 4,
    };

    const STEP: Duration = Duration::from_millis(7);

    /// A clock that advances `STEP` on every reading, so a hop takes exactly
    /// one step.
    fn stepping_clock() -> Clock {
        let readings = AtomicU64::new(0);
        Arc::new(move || STEP * readings.fetch_add(1, Ordering::Relaxed) as u32)
    }

    struct SourceState {
        engine_id: String,
        status: StatusCode,
        queries: usize,
        ids: Vec<String>,
    }

    type Shared = Arc<Mutex<SourceState>>;

    /// A vLLM worker running MooncakeConnector: an OpenAI server and a
    /// bootstrap registry on one listener.
    struct Source {
        state: Shared,
        served: Served,
    }

    async fn query(State(state): State<Shared>) -> axum::Json<Value> {
        let mut s = state.lock().unwrap();
        s.queries += 1;
        axum::Json(json!({"0": {"engine_id": s.engine_id}}))
    }

    async fn chat(State(state): State<Shared>, headers: HeaderMap) -> (StatusCode, String) {
        let mut s = state.lock().unwrap();
        let id = headers.get("X-Request-Id").and_then(|v| v.to_str().ok());
        s.ids.push(id.unwrap_or_default().to_owned());
        (
            s.status,
            r#"{"choices":[{"finish_reason":"length"}]}"#.to_owned(),
        )
    }

    impl Source {
        async fn start() -> Self {
            let state = Arc::new(Mutex::new(SourceState {
                engine_id: "engine-vllm-0".to_owned(),
                status: StatusCode::OK,
                queries: 0,
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

        /// Registry queries, and the request ids of the chat requests.
        fn seen(&self) -> (usize, Vec<String>) {
            let s = self.state.lock().unwrap();
            (s.queries, s.ids.clone())
        }

        fn endpoint(&self) -> Endpoint {
            endpoint(&self.served)
        }
    }

    fn endpoint(served: &Served) -> Endpoint {
        Endpoint {
            base_url: served.url(),
            bootstrap_url: served.url(),
        }
    }

    /// A source whose chat answer waits until `release` is set, and that
    /// sends on `entered` each time a chat request arrives.
    async fn stalling_source() -> (Served, watch::Sender<bool>, mpsc::UnboundedReceiver<()>) {
        let (release, released) = watch::channel(false);
        let (entered_tx, entered) = mpsc::unbounded_channel();
        let router = Router::new()
            .route("/query", get(|| async { r#"{"0":{"engine_id":"e0"}}"# }))
            .route(
                "/v1/chat/completions",
                post(move || {
                    let mut released = released.clone();
                    let entered = entered_tx.clone();
                    async move {
                        let _ = entered.send(());
                        let _ = released.wait_for(|r| *r).await;
                        r#"{"choices":[{"finish_reason":"length"}]}"#
                    }
                }),
            );
        let served = serve(router).await.expect("serve the stalling source");
        (served, release, entered)
    }

    fn hopper(cfg: Config, endpoints: &[(&str, Endpoint)]) -> Hopper {
        let endpoints = endpoints
            .iter()
            .map(|(pod, e)| ((*pod).to_owned(), e.clone()))
            .collect();
        Hopper::new(cfg, endpoints, reqwest::Client::new(), stepping_clock()).expect("Hopper::new")
    }

    fn source_hopper(src: &Source) -> Hopper {
        let unused = Endpoint {
            base_url: "http://unused".to_owned(),
            bootstrap_url: "http://unused".to_owned(),
        };
        hopper(FAST_RULE, &[("vllm-0", src.endpoint()), ("vllm-1", unused)])
    }

    fn mv(sticky: Sticky, history: u64) -> Move<'static> {
        Move {
            sticky,
            from: "vllm-0",
            to: "vllm-1",
            history,
        }
    }

    fn body() -> Bytes {
        Bytes::from_static(DOCTOR_STEP.as_bytes())
    }

    fn kv_params(out: &[u8]) -> Value {
        let got: Value = serde_json::from_slice(out).expect("forwarded body is not JSON");
        got["kv_transfer_params"].clone()
    }

    fn valid_transfer_id(id: &str) -> bool {
        id.strip_prefix("xfer-").is_some_and(|hex| {
            hex.len() == 32 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        })
    }

    #[tokio::test]
    async fn move_hops_a_long_history_off_the_old_worker() {
        let src = Source::start().await;
        let (out, report) = source_hopper(&src)
            .move_kv(&mv(Sticky::BrokenLoad, 5000), body())
            .await;
        assert!(
            matches!(report, Report::Hopped { took: STEP, .. }),
            "move_kv = {report:?}"
        );
        assert_eq!(report.outcome(), Some(Outcome::Hopped));
        assert!(report.error().is_none());
        let params = kv_params(&out);
        let xfer = params["transfer_id"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert_eq!(params["do_remote_prefill"], json!(true));
        assert_eq!(params["remote_engine_id"], json!("engine-vllm-0"));
        assert!(valid_transfer_id(&xfer), "transfer id {xfer:?}");
        assert_eq!(
            src.seen().1,
            [xfer],
            "want one source request under the transfer id"
        );
    }

    #[tokio::test]
    async fn move_leaves_the_body_alone_when_no_hop_applies() {
        let cases = [
            ("a kept binding", mv(Sticky::Hit, 5000), None),
            ("a first step", mv(Sticky::New, 5000), None),
            ("a restarted worker", mv(Sticky::BrokenGone, 5000), None),
            (
                "an unknown source",
                Move {
                    from: "vllm-9",
                    ..mv(Sticky::BrokenLoad, 5000)
                },
                None,
            ),
            (
                "the same worker",
                Move {
                    to: "vllm-0",
                    ..mv(Sticky::BrokenLoad, 5000)
                },
                None,
            ),
            (
                "a short history",
                mv(Sticky::BrokenShed, 5),
                Some(Outcome::BelowThreshold),
            ),
        ];
        for (name, m, want) in cases {
            let src = Source::start().await;
            let (out, report) = source_hopper(&src).move_kv(&m, body()).await;
            assert_eq!(report.outcome(), want, "{name}: {report:?}");
            assert_eq!(out, DOCTOR_STEP.as_bytes(), "{name}: body changed");
            assert_eq!(report.took(), Duration::ZERO, "{name}");
            assert_eq!(src.seen().1.len(), 0, "{name}: source got requests");
        }
    }

    #[tokio::test]
    async fn move_falls_back_to_recompute_when_the_source_fails() {
        let src = Source::start().await;
        src.set(|s| s.status = StatusCode::SERVICE_UNAVAILABLE);
        let (out, report) = source_hopper(&src)
            .move_kv(&mv(Sticky::BrokenLoad, 5000), body())
            .await;
        assert_eq!(report.outcome(), Some(Outcome::Failed), "{report:?}");
        assert_eq!(
            report.error().map(ToString::to_string).as_deref(),
            Some("source answered 503")
        );
        assert_eq!(out, DOCTOR_STEP.as_bytes());
    }

    #[tokio::test]
    async fn move_gives_up_on_a_source_slower_than_the_timeout() {
        let (slow, _release, _entered) = stalling_source().await;
        let cfg = Config {
            timeout: GoDuration::from_nanos(20_000_000),
            ..FAST_RULE
        };
        let h = hopper(cfg, &[("vllm-0", endpoint(&slow))]);
        let started = tokio::time::Instant::now();
        let (out, report) = h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await;
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "move_kv outlived its timeout"
        );
        assert_eq!(report.outcome(), Some(Outcome::Failed), "{report:?}");
        assert_eq!(
            report.error().map(ToString::to_string).as_deref(),
            Some("source did not hold the KV within 20ms")
        );
        assert_eq!(out, DOCTOR_STEP.as_bytes());
    }

    #[test]
    fn new_rejects_a_bad_config_and_no_endpoints() {
        let zero = Config {
            min_tokens: 0,
            shared_prefix_tokens: 0,
            kv_bytes_per_token: 0,
            transfer_bytes_per_s: 0.0,
            prefill_tokens_per_s: 0.0,
            overhead: GoDuration::from_nanos(0),
            timeout: GoDuration::from_nanos(0),
            receive_timeout: GoDuration::from_nanos(0),
            max_inflight: 0,
        };
        let one = HashMap::from([(
            "vllm-0".to_owned(),
            Endpoint {
                base_url: String::new(),
                bootstrap_url: String::new(),
            },
        )]);
        let bad = Hopper::new(zero, one, reqwest::Client::new(), stepping_clock());
        assert!(
            bad.as_ref().err().is_some_and(|err| err
                .to_string()
                .starts_with("hop config: KV bytes per token 0 must be positive\n")),
            "Hopper::new with a zero config"
        );
        let none = Hopper::new(
            FAST_RULE,
            HashMap::new(),
            reqwest::Client::new(),
            stepping_clock(),
        );
        assert_eq!(
            none.err().map(|err| err.to_string()).as_deref(),
            Some("hop needs at least one worker endpoint")
        );
    }

    #[tokio::test]
    async fn every_hop_gets_its_own_unguessable_transfer_id() {
        let src = Source::start().await;
        let h = source_hopper(&src);
        let mut seen = HashSet::new();
        for i in 1..=3 {
            let (out, report) = h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await;
            assert_eq!(report.outcome(), Some(Outcome::Hopped), "move_kv #{i}");
            let id = kv_params(&out)["transfer_id"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            assert!(
                valid_transfer_id(&id) && seen.insert(id.clone()),
                "transfer id {id:?} is malformed or repeated across hops {seen:?}"
            );
        }
    }

    #[tokio::test]
    async fn move_is_busy_past_max_inflight_and_frees_its_slot() {
        let (slow, release, mut entered) = stalling_source().await;
        let cfg = Config {
            max_inflight: 1,
            ..FAST_RULE
        };
        let h = Arc::new(hopper(cfg, &[("vllm-0", endpoint(&slow))]));
        let first = tokio::spawn({
            let h = Arc::clone(&h);
            async move { h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await.1 }
        });
        entered.recv().await.expect("first hop reached the source");

        let (out, report) = h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await;
        assert_eq!(report.outcome(), Some(Outcome::Busy), "second move_kv");
        assert_eq!(out, DOCTOR_STEP.as_bytes());
        release.send_replace(true);
        let first = first.await.expect("first move_kv panicked");
        assert_eq!(first.outcome(), Some(Outcome::Hopped), "first move_kv");

        let (_, report) = h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await;
        assert_eq!(
            report.outcome(),
            Some(Outcome::Hopped),
            "move_kv after the slot freed"
        );
    }

    #[tokio::test]
    async fn a_dropped_move_frees_its_slot() {
        let (slow, release, mut entered) = stalling_source().await;
        let cfg = Config {
            max_inflight: 1,
            ..FAST_RULE
        };
        let h = hopper(cfg, &[("vllm-0", endpoint(&slow))]);
        let m = mv(Sticky::BrokenLoad, 5000);
        let dropped = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::select! {
                _ = h.move_kv(&m, body()) => false,
                _ = entered.recv() => true,
            }
        })
        .await;
        assert_eq!(
            dropped,
            Ok(true),
            "the move reached the source and was dropped"
        );

        release.send_replace(true);
        let (_, report) = h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await;
        assert_eq!(
            report.outcome(),
            Some(Outcome::Hopped),
            "move_kv after a dropped move"
        );
    }

    #[tokio::test]
    async fn forget_makes_the_next_hop_ask_the_registry_again() {
        let src = Source::start().await;
        let h = source_hopper(&src);
        for i in 1..=2 {
            let (_, report) = h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await;
            assert_eq!(report.outcome(), Some(Outcome::Hopped), "move_kv #{i}");
        }
        assert_eq!(src.seen().0, 1, "registry queries for two hops");

        h.forget("vllm-9");
        h.forget("vllm-0");
        src.set(|s| s.engine_id = "engine-vllm-0-restarted".to_owned());
        let (out, report) = h.move_kv(&mv(Sticky::BrokenLoad, 5000), body()).await;
        assert_eq!(
            report.outcome(),
            Some(Outcome::Hopped),
            "move_kv after forget"
        );
        assert_eq!(
            kv_params(&out)["remote_engine_id"],
            json!("engine-vllm-0-restarted")
        );
        assert_eq!(src.seen().0, 2, "want a second registry query after forget");
        assert_eq!(h.receive_timeout(), Duration::from_secs(1));
    }
}
