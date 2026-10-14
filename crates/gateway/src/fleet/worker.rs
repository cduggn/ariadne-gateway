use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use ariadne_decide::Timestamp;
use bytes::Bytes;
use reqwest::StatusCode;
use tokio::task::JoinSet;
use tokio::time::{MissedTickBehavior, timeout};
use tokio_util::sync::CancellationToken;

use super::gate::{Gate, Scrape};
use super::lock;
use super::scrape::{Histogram, MAX_METRICS_BYTES, ParseError, Sample, parse_metrics};
use crate::clock::Clock;

/// Where a worker is in its life as the fleet sees it. A worker starts Down, becomes Warming on
/// its first successful scrape, Ready once warm-up passes, and Down again after `down_after`
/// without a scrape. The Gate's ready flag follows it: set on Ready, cleared on Down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Down,
    Warming,
    Ready,
}

impl Phase {
    /// The `orch_replica_phase` label and the `/debug/workers` text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Down => "down",
            Self::Warming => "warming",
            Self::Ready => "ready",
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The shape of every per-pod loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    /// The scrape period.
    pub interval: Duration,
    pub scrape_timeout: Duration,
    /// How long a worker may go without a successful scrape before it is Down.
    pub down_after: Duration,
    /// How far back the TTFT quantiles look.
    pub window: Duration,
    /// How far back `waiting_rising` compares.
    pub rising_over: Duration,
    /// The recorded step-1 request a warm-up probe replays.
    pub warm_body: Bytes,
    /// A probe passes when it answers 200 in less than this.
    pub warm_under: Duration,
    /// Consecutive passes that make a worker Ready.
    pub warm_passes: usize,
    pub probe_timeout: Duration,
}

impl WorkerConfig {
    /// The shipped loop: scrape twice a second, Down after 10 s of silence, TTFT over the last
    /// 30 s, and two probes under 1 s before a worker is Ready.
    pub fn defaults(warm_body: Bytes) -> Self {
        Self {
            interval: Duration::from_millis(500),
            scrape_timeout: Duration::from_secs(1),
            down_after: Duration::from_secs(10),
            window: Duration::from_secs(30),
            rising_over: Duration::from_secs(5),
            warm_body,
            warm_under: Duration::from_secs(1),
            warm_passes: 2,
            probe_timeout: Duration::from_secs(30),
        }
    }
}

/// One worker as the fleet last saw it, for metrics and `/debug`. `last_scrape` is `None` before
/// the first successful scrape. `restarts` counts Down to Warming transitions, including the
/// first one at startup.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerStatus {
    pub pod: String,
    pub phase: Phase,
    pub last_scrape: Option<SystemTime>,
    /// The latest warm-up probe's latency in seconds.
    pub last_probe_s: f64,
    pub probes: u64,
    pub restarts: u64,
}

/// Why a fetch from a worker failed.
#[derive(Debug)]
pub enum FetchError {
    UnknownPod(String),
    Timeout {
        url: String,
    },
    Transport(reqwest::Error),
    Status {
        url: String,
        status: StatusCode,
    },
    Parse(ParseError),
    /// The body had no KV usage, so the Gate could not place on the worker.
    NoKvUsage,
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPod(pod) => write!(f, "fleet: unknown pod {pod:?}"),
            Self::Timeout { url } => write!(f, "Get {url:?}: context deadline exceeded"),
            Self::Transport(e) => e.fmt(f),
            Self::Status { url, status } => write!(f, "GET {url}: {status}"),
            Self::Parse(e) => e.fmt(f),
            Self::NoKvUsage => f.write_str("metrics body has no KV cache usage"),
        }
    }
}

impl std::error::Error for FetchError {}

impl From<reqwest::Error> for FetchError {
    fn from(e: reqwest::Error) -> Self {
        Self::Transport(e)
    }
}

/// Reads at most `cap` bytes of `resp`'s body, chunk by chunk, and drops the rest unread. Every
/// read of an upstream body goes through here, so a misbehaving upstream cannot make the gateway
/// buffer more than `cap`.
pub async fn read_capped(
    mut resp: reqwest::Response,
    cap: usize,
) -> Result<Vec<u8>, reqwest::Error> {
    let mut body = Vec::new();
    while body.len() < cap {
        let Some(chunk) = resp.chunk().await? else {
            break;
        };
        let take = chunk.len().min(cap - body.len());
        body.extend_from_slice(&chunk[..take]);
    }
    Ok(body)
}

/// The part of a scrape the fleet uses, with KV usage known to be present.
#[derive(Debug, Clone, Default, PartialEq)]
struct Metrics {
    kv_usage: f64,
    kv_pool_tokens: u64,
    waiting: f64,
    ttft: Histogram,
}

impl Metrics {
    fn from_sample(s: Sample) -> Option<Self> {
        Some(Self {
            kv_usage: s.kv_usage?,
            kv_pool_tokens: s.kv_pool_tokens,
            waiting: s.waiting,
            ttft: s.ttft,
        })
    }
}

/// One scrape and when it was read.
#[derive(Debug, Clone, PartialEq)]
struct Reading {
    at: Timestamp,
    metrics: Metrics,
}

/// The scrapes of the last `window`, never empty: the newest plus the older ones, oldest first.
/// One reading at or before the window's start stays, so the TTFT delta spans the whole window.
#[derive(Debug)]
struct Ring {
    older: VecDeque<Reading>,
    newest: Reading,
}

impl Ring {
    fn new(first: Reading) -> Self {
        Self {
            older: VecDeque::new(),
            newest: first,
        }
    }

    fn push(&mut self, r: Reading, window: Duration) {
        let prev = std::mem::replace(&mut self.newest, r);
        self.older.push_back(prev);
        let now = self.newest.at;
        while !self.older.is_empty() {
            let second = self.older.get(1).unwrap_or(&self.newest);
            if second.at.saturating_add(window) > now {
                break;
            }
            self.older.pop_front();
        }
    }

    /// The Gate's view of the newest reading. The TTFT quantiles cover the delta between the
    /// oldest and the newest reading, so the first scrape after a restart quotes 0.
    /// `waiting_rising` compares the newest reading with the newest one at least `rising_over`
    /// old, and is false while the ring is too short to tell.
    fn scrape(&self, rising_over: Duration) -> Scrape {
        let cur = &self.newest;
        let oldest = self.older.front().unwrap_or(cur);
        let rising = std::iter::once(cur)
            .chain(self.older.iter().rev())
            .find(|r| r.at.saturating_add(rising_over) <= cur.at)
            .is_some_and(|base| cur.metrics.waiting > base.metrics.waiting);
        let ttft = &cur.metrics.ttft;
        Scrape {
            at: cur.at,
            kv_pool_tokens: cur.metrics.kv_pool_tokens,
            kv_usage: cur.metrics.kv_usage,
            ttft_p50_s: ttft.quantile(&oldest.metrics.ttft, 0.5),
            ttft_p99_s: ttft.quantile(&oldest.metrics.ttft, 0.99),
            waiting_rising: rising,
        }
    }
}

/// What moved a worker: a successful scrape, `warm_passes` consecutive probe passes, or
/// `down_after` without a successful scrape.
#[derive(Debug)]
enum Event {
    Scraped(Reading),
    Warmed,
    Stale,
}

/// One worker's loop state. The ring lives only while the worker is up, and is new on every
/// restart because vLLM's counters start over.
#[derive(Debug, Default)]
enum Life {
    #[default]
    Down,
    Warming {
        ring: Ring,
        passes: usize,
    },
    Ready {
        ring: Ring,
    },
}

impl Life {
    fn phase(&self) -> Phase {
        match self {
            Self::Down => Phase::Down,
            Self::Warming { .. } => Phase::Warming,
            Self::Ready { .. } => Phase::Ready,
        }
    }

    /// The transition table. Every pair not listed keeps its phase, so a stale Down worker or a
    /// warmed Ready worker is a no-op.
    fn step(self, e: Event, window: Duration) -> Self {
        match (self, e) {
            (Self::Down, Event::Scraped(r)) => Self::Warming {
                ring: Ring::new(r),
                passes: 0,
            },
            (Self::Warming { mut ring, passes }, Event::Scraped(r)) => {
                ring.push(r, window);
                Self::Warming { ring, passes }
            }
            (Self::Ready { mut ring }, Event::Scraped(r)) => {
                ring.push(r, window);
                Self::Ready { ring }
            }
            (Self::Warming { ring, .. }, Event::Warmed) => Self::Ready { ring },
            (Self::Warming { .. } | Self::Ready { .. }, Event::Stale) => Self::Down,
            (life @ (Self::Down | Self::Ready { .. }), Event::Warmed)
            | (life @ Self::Down, Event::Stale) => life,
        }
    }

    fn ring(&self) -> Option<&Ring> {
        match self {
            Self::Down => None,
            Self::Warming { ring, .. } | Self::Ready { ring } => Some(ring),
        }
    }
}

struct Member {
    name: String,
    base: String,
    status: Mutex<WorkerStatus>,
}

impl Member {
    fn update<T>(&self, f: impl FnOnce(&mut WorkerStatus) -> T) -> T {
        f(&mut lock(&self.status))
    }
}

type Hook = Arc<dyn Fn(&str) + Send + Sync>;

struct Inner {
    gate: Gate,
    cfg: WorkerConfig,
    client: reqwest::Client,
    clock: Clock,
    members: Vec<Member>,
    not_ready: Mutex<Option<Hook>>,
}

/// Keeps the Gate fed: one loop per pod scrapes its `/metrics`, feeds each scrape to the Gate
/// and drives the pod through the warm-up gate. The Gate never calls back into the Fleet.
/// Cloning is cheap and every clone is the same Fleet.
#[derive(Clone)]
pub struct Fleet(Arc<Inner>);

impl fmt::Debug for Fleet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fleet")
            .field("status", &self.status())
            .finish()
    }
}

impl Fleet {
    /// A Fleet over `urls`, which maps each pod to its base URL.
    pub fn new(
        gate: Gate,
        urls: BTreeMap<String, String>,
        cfg: WorkerConfig,
        client: reqwest::Client,
        clock: Clock,
    ) -> Self {
        let members = urls
            .into_iter()
            .map(|(name, base)| Member {
                base: base.strip_suffix('/').unwrap_or(&base).to_owned(),
                status: Mutex::new(WorkerStatus {
                    pod: name.clone(),
                    phase: Phase::Down,
                    last_scrape: None,
                    last_probe_s: 0.0,
                    probes: 0,
                    restarts: 0,
                }),
                name,
            })
            .collect();
        Self(Arc::new(Inner {
            gate,
            cfg,
            client,
            clock,
            members,
            not_ready: Mutex::new(None),
        }))
    }

    /// Runs `hook` each time a worker enters a phase other than Ready: Down on an outage and
    /// Warming on a restart. Anything cached about the worker's engine, such as the KV hop's
    /// engine id, is stale from then on. A later call replaces the hook.
    pub fn on_not_ready(&self, hook: impl Fn(&str) + Send + Sync + 'static) {
        *lock(&self.0.not_ready) = Some(Arc::new(hook));
    }

    /// Runs one loop per pod until `shutdown` is cancelled and every loop has stopped.
    pub async fn run(&self, shutdown: CancellationToken) {
        let mut loops = JoinSet::new();
        for i in 0..self.0.members.len() {
            let (fleet, shutdown) = (self.clone(), shutdown.clone());
            loops.spawn(async move { fleet.watch(i, shutdown).await });
        }
        while loops.join_next().await.is_some() {}
    }

    /// Every worker's status, in pod name order.
    pub fn status(&self) -> Vec<WorkerStatus> {
        self.0
            .members
            .iter()
            .map(|m| m.update(|st| st.clone()))
            .collect()
    }

    /// A pod's `/metrics` body as the pod sent it, capped at `MAX_METRICS_BYTES`.
    pub async fn raw_metrics(&self, pod: &str) -> Result<Vec<u8>, FetchError> {
        let m = self
            .0
            .members
            .iter()
            .find(|m| m.name == pod)
            .ok_or_else(|| FetchError::UnknownPod(pod.to_owned()))?;
        self.get_metrics(m).await
    }

    /// One pod's loop. The first tick runs at once.
    async fn watch(&self, i: usize, shutdown: CancellationToken) {
        let Some(m) = self.0.members.get(i) else {
            return;
        };
        // tokio panics on a zero period.
        let mut ticker = tokio::time::interval(self.0.cfg.interval.max(Duration::from_millis(1)));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        ticker.tick().await;
        let mut life = Life::Down;
        loop {
            tokio::select! {
                () = shutdown.cancelled() => return,
                () = self.tick(m, &mut life) => {}
            }
            tokio::select! {
                () = shutdown.cancelled() => return,
                _ = ticker.tick() => {}
            }
        }
    }

    /// Scrapes once and applies what it learned. A failed scrape only ages the worker; a
    /// successful one feeds the Gate, then probes while the worker is Warming.
    async fn tick(&self, m: &Member, life: &mut Life) {
        let g = &*self.0;
        let read = self.scrape(m).await;
        let now = g.clock.now();
        let metrics = match read {
            Ok(metrics) => metrics,
            Err(_) => {
                let stale = life
                    .ring()
                    .is_some_and(|r| r.newest.at.saturating_add(g.cfg.down_after) <= now);
                if stale {
                    self.advance(m, life, Event::Stale);
                }
                return;
            }
        };

        self.advance(m, life, Event::Scraped(Reading { at: now, metrics }));
        m.update(|st| st.last_scrape = Some(g.clock.wall(now)));
        if let Some(ring) = life.ring() {
            g.gate.observe(&m.name, ring.scrape(g.cfg.rising_over));
        }

        let warmed = match life {
            Life::Warming { passes, .. } => {
                *passes = if self.probe(m).await {
                    passes.saturating_add(1)
                } else {
                    0
                };
                *passes >= g.cfg.warm_passes
            }
            Life::Down | Life::Ready { .. } => false,
        };
        if warmed {
            self.advance(m, life, Event::Warmed);
        }
    }

    /// Applies one event and does what entering the new phase owes the Gate. Entering Warming
    /// is a restart, so the Gate drops the worker's bindings. Entering Ready or Down sets or
    /// clears the Gate's ready flag. Entering Warming or Down runs the `on_not_ready` hook.
    fn advance(&self, m: &Member, life: &mut Life, e: Event) {
        let g = &*self.0;
        let before = life.phase();
        *life = std::mem::take(life).step(e, g.cfg.window);
        let next = life.phase();
        if next == before {
            return;
        }
        m.update(|st| {
            st.phase = next;
            if next == Phase::Warming {
                st.restarts = st.restarts.saturating_add(1);
            }
        });
        match next {
            Phase::Warming => g.gate.restarted(&m.name),
            Phase::Ready => g.gate.set_ready(&m.name, true),
            Phase::Down => g.gate.set_ready(&m.name, false),
        }
        if next != Phase::Ready {
            let hook = lock(&g.not_ready).clone();
            if let Some(hook) = hook {
                hook(&m.name);
            }
        }
    }

    async fn scrape(&self, m: &Member) -> Result<Metrics, FetchError> {
        let body = self.get_metrics(m).await?;
        let sample = parse_metrics(&body).map_err(FetchError::Parse)?;
        Metrics::from_sample(sample).ok_or(FetchError::NoKvUsage)
    }

    /// Fetches `/metrics` within `scrape_timeout`. Any status but 200 is an error.
    async fn get_metrics(&self, m: &Member) -> Result<Vec<u8>, FetchError> {
        let url = format!("{}/metrics", m.base);
        let get = async {
            let resp = self.0.client.get(&url).send().await?;
            if resp.status() != StatusCode::OK {
                return Err(FetchError::Status {
                    url: url.clone(),
                    status: resp.status(),
                });
            }
            Ok(read_capped(resp, MAX_METRICS_BYTES).await?)
        };
        timeout(self.0.cfg.scrape_timeout, get)
            .await
            .unwrap_or_else(|_| Err(FetchError::Timeout { url: url.clone() }))
    }

    /// Replays `warm_body` once and scores it: a 200 under `warm_under` passes, and anything
    /// else, a transport error included, fails. The request id carries the pod and the probe
    /// number, so vLLM's logs and the Gate's run table can tell probes from live traffic.
    async fn probe(&self, m: &Member) -> bool {
        let g = &*self.0;
        let n = m.update(|st| {
            st.probes = st.probes.saturating_add(1);
            st.probes
        });
        let post = async {
            let resp = g
                .client
                .post(format!("{}/v1/chat/completions", m.base))
                .header("Content-Type", "application/json")
                .header("X-Request-Id", format!("warmup-{}-{n}-s1", m.name))
                .body(g.cfg.warm_body.clone())
                .send()
                .await?;
            let status = resp.status();
            // Drained so the client can reuse the connection; only the status matters.
            read_capped(resp, MAX_METRICS_BYTES).await?;
            Ok::<_, reqwest::Error>(status)
        };
        let start = g.clock.now();
        let status = timeout(g.cfg.probe_timeout, post).await;
        let elapsed = g.clock.now().saturating_duration_since(start);
        m.update(|st| st.last_probe_s = elapsed.as_secs_f64());
        matches!(status, Ok(Ok(StatusCode::OK))) && elapsed < g.cfg.warm_under
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::Router;
    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::routing::{get, post};
    use tokio_util::sync::DropGuard;

    use super::super::gate::tests::{must_admit, request, view_of, zero_rnd};
    use super::super::gate::{Config, Outcome};
    use super::super::scrape::tests::VLLM_METRICS;
    use super::*;
    use ariadne_decide::{RunId, RunUsage};

    const POD: &str = "vllm-0";
    const WARM_BODY: &[u8] = br#"{"model":"Qwen/Qwen3-8B-AWQ","messages":[{"role":"user","content":"warm"}],"max_completion_tokens":1}"#;

    struct Knobs {
        metrics: String,
        metrics_fail: bool,
        probe_delay: Duration,
        probe_status: StatusCode,
        probe_bodies: Vec<Bytes>,
        probe_ids: Vec<String>,
    }

    type Shared = Arc<Mutex<Knobs>>;

    /// Serves `/metrics` and `/v1/chat/completions` the way the fleet expects, with every knob
    /// settable mid-test.
    struct FakeVllm {
        url: String,
        knobs: Shared,
        server: tokio::task::JoinHandle<()>,
    }

    impl Drop for FakeVllm {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    async fn metrics(State(k): State<Shared>) -> (StatusCode, String) {
        let k = k.lock().expect("knobs");
        if k.metrics_fail {
            (StatusCode::SERVICE_UNAVAILABLE, "down\n".to_owned())
        } else {
            (StatusCode::OK, k.metrics.clone())
        }
    }

    async fn completions(
        State(k): State<Shared>,
        headers: HeaderMap,
        body: Bytes,
    ) -> (StatusCode, &'static str) {
        let (delay, status) = {
            let mut k = k.lock().expect("knobs");
            k.probe_bodies.push(body);
            let id = headers.get("x-request-id").and_then(|v| v.to_str().ok());
            k.probe_ids.push(id.unwrap_or_default().to_owned());
            (k.probe_delay, k.probe_status)
        };
        tokio::time::sleep(delay).await;
        (status, r#"{"choices":[]}"#)
    }

    impl FakeVllm {
        async fn start() -> Self {
            let knobs = Arc::new(Mutex::new(Knobs {
                metrics: VLLM_METRICS.to_owned(),
                metrics_fail: false,
                probe_delay: Duration::ZERO,
                probe_status: StatusCode::OK,
                probe_bodies: Vec::new(),
                probe_ids: Vec::new(),
            }));
            let app = Router::new()
                .route("/metrics", get(metrics))
                .route("/v1/chat/completions", post(completions))
                .with_state(Arc::clone(&knobs));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            let url = format!("http://{}", listener.local_addr().expect("addr"));
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.expect("serve");
            });
            Self { url, knobs, server }
        }

        fn set(&self, f: impl FnOnce(&mut Knobs)) {
            f(&mut self.knobs.lock().expect("knobs"));
        }

        fn probes(&self) -> (Vec<Bytes>, Vec<String>) {
            let k = self.knobs.lock().expect("knobs");
            (k.probe_bodies.clone(), k.probe_ids.clone())
        }
    }

    fn fast_worker_config() -> WorkerConfig {
        WorkerConfig {
            interval: Duration::from_millis(5),
            scrape_timeout: Duration::from_millis(200),
            down_after: Duration::from_millis(60),
            window: Duration::from_secs(1),
            rising_over: Duration::from_millis(20),
            warm_body: Bytes::from_static(WARM_BODY),
            warm_under: Duration::from_millis(50),
            warm_passes: 2,
            probe_timeout: Duration::from_secs(1),
        }
    }

    struct Running {
        gate: Gate,
        fleet: Fleet,
        clock: Clock,
        _stop: DropGuard,
    }

    fn new_fleet(fake: &FakeVllm) -> (Gate, Fleet, Clock) {
        let clock = Clock::system();
        let gate = Gate::new(
            Config::defaults(vec![POD.to_owned()]),
            clock.clone(),
            zero_rnd,
        );
        let urls = BTreeMap::from([(POD.to_owned(), fake.url.clone())]);
        let fleet = Fleet::new(
            gate.clone(),
            urls,
            fast_worker_config(),
            reqwest::Client::new(),
            clock.clone(),
        );
        (gate, fleet, clock)
    }

    fn start(gate: Gate, fleet: Fleet, clock: Clock) -> Running {
        let token = CancellationToken::new();
        tokio::spawn({
            let (fleet, token) = (fleet.clone(), token.clone());
            async move { fleet.run(token).await }
        });
        Running {
            gate,
            fleet,
            clock,
            _stop: token.drop_guard(),
        }
    }

    async fn eventually(what: &str, cond: impl Fn() -> bool) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while tokio::time::Instant::now() < deadline {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        panic!("timed out waiting for {what}");
    }

    fn status(f: &Fleet) -> WorkerStatus {
        f.status().remove(0)
    }

    fn ready_of(g: &Gate) -> bool {
        g.view().iter().any(|w| w.view.pod == POD && w.ready)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_warms_up_then_is_ready() {
        let fake = FakeVllm::start().await;
        let (gate, fleet, clock) = new_fleet(&fake);
        assert!(!ready_of(&gate), "pod ready before any scrape");
        let r = start(gate, fleet, clock);

        eventually("phase ready", || status(&r.fleet).phase == Phase::Ready).await;
        let st = status(&r.fleet);
        assert_eq!((st.pod.as_str(), st.restarts), (POD, 1), "{st:?}");
        assert!(st.probes >= 2 && st.last_scrape.is_some(), "{st:?}");
        assert!(
            st.last_probe_s > 0.0 && st.last_probe_s < 0.05,
            "last_probe_s = {}, want a fast probe under 0.05",
            st.last_probe_s
        );
        assert!(ready_of(&r.gate), "Gate view ready after warm-up");
        let v = view_of(&r.gate, POD);
        assert_eq!((v.kv_pool_tokens, v.kv_usage), (79056, 0.42));

        let (bodies, ids) = fake.probes();
        assert!(bodies.len() >= 2, "fake saw {} probes", bodies.len());
        assert!(
            bodies.iter().all(|b| b.as_ref() == WARM_BODY),
            "probe bodies"
        );
        assert_eq!(ids[..2], ["warmup-vllm-0-1-s1", "warmup-vllm-0-2-s1"]);

        let raw = r.fleet.raw_metrics(POD).await.expect("raw metrics");
        assert_eq!(String::from_utf8(raw).ok().as_deref(), Some(VLLM_METRICS));
        let err = r.fleet.raw_metrics("nope").await.expect_err("unknown pod");
        assert_eq!(err.to_string(), r#"fleet: unknown pod "nope""#);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_or_failing_probe_keeps_worker_warming() {
        let fake = FakeVllm::start().await;
        fake.set(|k| k.probe_delay = Duration::from_millis(80));
        let (gate, fleet, clock) = new_fleet(&fake);
        let r = start(gate, fleet, clock);

        eventually("two slow probes", || status(&r.fleet).probes >= 2).await;
        let st = status(&r.fleet);
        assert_eq!(st.phase, Phase::Warming, "after slow probes");
        assert!(
            st.last_probe_s >= 0.08,
            "last_probe_s = {}",
            st.last_probe_s
        );
        assert!(!ready_of(&r.gate), "Gate view ready after slow probes");

        fake.set(|k| {
            k.probe_delay = Duration::ZERO;
            k.probe_status = StatusCode::INTERNAL_SERVER_ERROR;
        });
        let seen = status(&r.fleet).probes;
        eventually("two failing probes", || status(&r.fleet).probes >= seen + 2).await;
        assert_eq!(status(&r.fleet).phase, Phase::Warming, "after 500 probes");

        fake.set(|k| k.probe_status = StatusCode::OK);
        eventually("phase ready once probes pass", || {
            status(&r.fleet).phase == Phase::Ready
        })
        .await;
        assert!(ready_of(&r.gate), "Gate view ready after probes pass");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn outage_marks_down_and_recovery_restarts() {
        let fake = FakeVllm::start().await;
        let (gate, fleet, clock) = new_fleet(&fake);
        let r = start(gate, fleet, clock);
        eventually("phase ready", || status(&r.fleet).phase == Phase::Ready).await;

        let run1 = RunId::new("run1").expect("run id");
        let (_, t) = must_admit(
            &r.gate,
            &request(r.clock.now(), "run1-s1", "platform", 7000, 768),
        )
        .await;
        t.settle(Outcome::Answered {
            usage: RunUsage {
                prompt_tokens: 9000,
                completion_tokens: 60,
                body_bytes: 7000,
            },
            latency: Duration::from_millis(300),
        });
        assert_eq!(r.gate.bound(&run1).as_deref(), Some(POD));

        fake.set(|k| k.metrics_fail = true);
        eventually("phase down", || status(&r.fleet).phase == Phase::Down).await;
        assert!(!ready_of(&r.gate), "Gate view ready while down");
        assert_eq!(
            r.gate.bound(&run1).as_deref(),
            Some(POD),
            "bound during the outage until a restart"
        );

        fake.set(|k| k.metrics_fail = false);
        eventually("restart counted", || status(&r.fleet).restarts == 2).await;
        assert_ne!(status(&r.fleet).phase, Phase::Down, "after recovery");
        assert_eq!(r.gate.bound(&run1), None, "after restart");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_not_ready_runs_on_outage_and_on_restart() {
        let fake = FakeVllm::start().await;
        let (gate, fleet, clock) = new_fleet(&fake);
        let calls = Arc::new(Mutex::new(Vec::new()));
        fleet.on_not_ready({
            let (fleet, calls) = (fleet.clone(), Arc::clone(&calls));
            move |pod| {
                let phase = status(&fleet).phase;
                calls.lock().expect("calls").push(format!("{pod}:{phase}"));
            }
        });
        let seen = || calls.lock().expect("calls").clone();
        let r = start(gate, fleet, clock);

        eventually("phase ready", || status(&r.fleet).phase == Phase::Ready).await;
        assert_eq!(seen(), ["vllm-0:warming"], "hook calls before the outage");

        fake.set(|k| k.metrics_fail = true);
        eventually("phase down", || status(&r.fleet).phase == Phase::Down).await;
        fake.set(|k| k.metrics_fail = false);
        eventually("phase ready again", || {
            status(&r.fleet).phase == Phase::Ready
        })
        .await;
        assert_eq!(seen(), ["vllm-0:warming", "vllm-0:down", "vllm-0:warming"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn waiting_rising_compares_against_rising_over_ago() {
        let fake = FakeVllm::start().await;
        let (gate, fleet, clock) = new_fleet(&fake);
        let r = start(gate, fleet, clock);
        eventually("phase ready", || status(&r.fleet).phase == Phase::Ready).await;
        assert!(
            !view_of(&r.gate, POD).waiting_rising,
            "on a flat waiting count"
        );

        fake.set(|k| {
            k.metrics = VLLM_METRICS.replacen(
                r#"vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 3.0"#,
                r#"vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 9.0"#,
                1,
            );
        });
        eventually("waiting_rising", || view_of(&r.gate, POD).waiting_rising).await;
        eventually("waiting_rising to settle", || {
            !view_of(&r.gate, POD).waiting_rising
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_capped_stops_at_the_cap() {
        let fake = FakeVllm::start().await;
        let resp = reqwest::get(format!("{}/metrics", fake.url))
            .await
            .expect("get");
        let body = read_capped(resp, 10).await.expect("read");
        assert_eq!(body, VLLM_METRICS.as_bytes()[..10]);
    }

    fn reading(secs: u64) -> Reading {
        Reading {
            at: Timestamp::from_epoch_offset(Duration::from_secs(secs)),
            metrics: Metrics::default(),
        }
    }

    fn life(phase: Phase) -> Life {
        let ring = || Ring::new(reading(0));
        match phase {
            Phase::Down => Life::Down,
            Phase::Warming => Life::Warming {
                ring: ring(),
                passes: 0,
            },
            Phase::Ready => Life::Ready { ring: ring() },
        }
    }

    #[test]
    fn step_transition_table() {
        use Phase::{Down, Ready, Warming};
        let scraped = || Event::Scraped(reading(1));
        let cases = [
            (Down, scraped(), Warming),
            (Down, Event::Warmed, Down),
            (Down, Event::Stale, Down),
            (Warming, scraped(), Warming),
            (Warming, Event::Warmed, Ready),
            (Warming, Event::Stale, Down),
            (Ready, scraped(), Ready),
            (Ready, Event::Warmed, Ready),
            (Ready, Event::Stale, Down),
        ];
        for (from, e, want) in cases {
            let label = format!("step({from}, {e:?})");
            assert_eq!(
                life(from).step(e, Duration::from_secs(1)).phase(),
                want,
                "{label}"
            );
        }
    }

    #[test]
    fn ring_keeps_one_reading_before_the_window() {
        let mut ring = Ring::new(reading(0));
        for s in [10, 20, 30, 40] {
            ring.push(reading(s), Duration::from_secs(25));
        }
        let kept: Vec<u64> = ring
            .older
            .iter()
            .chain(std::iter::once(&ring.newest))
            .map(|r| r.at.epoch_offset().as_secs())
            .collect();
        assert_eq!(kept, [10, 20, 30, 40]);
    }

    #[test]
    fn phase_string() {
        assert_eq!(
            [Phase::Down, Phase::Warming, Phase::Ready].map(Phase::as_str),
            ["down", "warming", "ready"]
        );
        assert_eq!(Phase::Warming.to_string(), "warming");
    }
}
