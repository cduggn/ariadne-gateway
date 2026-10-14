use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ariadne_decide::{
    Budgets, DEFAULT_BUDGETS, DEFAULT_POLICY, DEFAULT_STALENESS, DEFAULT_STICK_SLACK, Fleet,
    PickPolicy, Placement, Policy, Reason, Refusal, Request, RunId, RunUsage, Staleness,
    TenantState, Timestamp, WorkerState, WorkerView, estimate_tokens, pick,
};

use super::lock;
use super::queue::{Queue, Slot};
use crate::clock::Clock;

/// The tenant key whose bucket every tenant not listed in `Config::tenants` shares.
pub const SHARED_TENANT: &str = "*";

/// One tenant's token bucket shape: its capacity and its refill rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TenantLimit {
    pub rate_per_s: f64,
    pub burst: f64,
}

/// Everything the Gate needs to decide, reserve and queue.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The fixed worker set, in the order `Gate::view` reports it.
    pub pods: Vec<String>,
    pub max_inflight: usize,
    pub max_queued: usize,
    pub policy: PickPolicy,
    pub shed: Policy,
    pub staleness: Staleness,
    pub stick_slack: f64,
    pub budgets: Budgets,
    /// Per-tenant limits. `SHARED_TENANT` is the bucket for every tenant not listed; without it
    /// those tenants share an empty bucket and are refused.
    pub tenants: HashMap<String, TenantLimit>,
    /// How far before a scrape a reservation still counts, because the scrape may not yet
    /// reflect a request dispatched just before it.
    pub reserve_grace: Duration,
    /// How long an idle run keeps its binding.
    pub run_ttl: Duration,
}

/// The largest context plus output.
const MAX_REQUEST_TOKENS: u64 = 32768;

impl Config {
    /// The shipped configuration. The platform tenant is the doctor itself, and its bucket is
    /// sized never to bind before the KV and queue gates: its burst fills every in-flight slot
    /// at the largest request size, and it refills in 4 s. The charge counts cached prompt
    /// tokens, so chained steps spend far more than their new tokens. Every other tenant shares
    /// one bucket of one largest request refilled at 833.33 tokens/s, which is where the 429
    /// path shows.
    pub fn defaults(pods: Vec<String>) -> Self {
        let max_inflight = 16;
        let platform_burst = (pods.len() as u64)
            .saturating_mul(max_inflight)
            .saturating_mul(MAX_REQUEST_TOKENS) as f64;
        Self {
            pods,
            max_inflight: max_inflight as usize,
            max_queued: 32,
            policy: PickPolicy::PrefixThenLoad,
            shed: DEFAULT_POLICY,
            staleness: DEFAULT_STALENESS,
            stick_slack: DEFAULT_STICK_SLACK,
            budgets: DEFAULT_BUDGETS,
            tenants: HashMap::from([
                (
                    "platform".to_owned(),
                    TenantLimit {
                        rate_per_s: platform_burst / 4.0,
                        burst: platform_burst,
                    },
                ),
                (
                    SHARED_TENANT.to_owned(),
                    TenantLimit {
                        rate_per_s: 833.33,
                        burst: MAX_REQUEST_TOKENS as f64,
                    },
                ),
            ]),
            reserve_grace: Duration::from_secs(1),
            run_ttl: Duration::from_secs(600),
        }
    }
}

/// One worker's telemetry as the scraper read it. `at` is the time of the read, which sets the
/// worker's age and which reservations the scrape already reflects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scrape {
    pub at: Timestamp,
    pub kv_pool_tokens: u64,
    pub kv_usage: f64,
    pub ttft_p50_s: f64,
    pub ttft_p99_s: f64,
    pub waiting_rising: bool,
}

/// What the handler learned from the worker or the overflow backend.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Outcome {
    /// No usable answer: the charge is refunded in full.
    #[default]
    Failed,
    /// `usage.body_bytes` is the request body's length, which the run's next estimate builds on.
    Answered { usage: RunUsage, latency: Duration },
}

/// `admit`'s view of one decision. `est` is the token estimate it used, `queued` the time spent
/// waiting for a slot. `from` is the worker the run was bound to before this decision and
/// `history` the prompt length of its last settled step: together they say what a moved run
/// left behind, which a KV hop can copy.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub placement: Placement,
    pub est: u64,
    pub queued: Duration,
    pub from: Option<String>,
    pub history: u64,
}

/// `admit`'s result. `Admitted` carries the placed decision and the hold on the fleet;
/// `Refused` carries a decision whose placement is an `Err`.
#[derive(Debug)]
pub enum Admission {
    Admitted(Decision, Ticket),
    Refused(Decision),
}

impl Admission {
    pub fn decision(&self) -> &Decision {
        match self {
            Self::Admitted(d, _) | Self::Refused(d) => d,
        }
    }
}

/// What one request was charged against its tenant, and the run its usage belongs to.
#[derive(Debug)]
struct Charge {
    tenant: String,
    run: Option<RunId>,
    est: u64,
}

/// A charge and a reservation on `pod`. Dropping it settles both with `outcome`.
struct Hold {
    gate: Arc<Shared>,
    charge: Charge,
    pod: String,
    reservation: u64,
    outcome: Outcome,
}

impl Drop for Hold {
    fn drop(&mut self) {
        let g = &*self.gate;
        let mut st = lock(&g.state);
        let now = g.clock.now();
        if let Some(p) = st.pod_mut(&self.pod) {
            p.reservations.retain(|r| r.id != self.reservation);
            if let Outcome::Answered { latency, .. } = self.outcome {
                p.observe_service(latency);
            }
        }
        st.settle_charge(now, &self.charge, &self.outcome);
    }
}

/// One admitted request's hold on the fleet: its tenant charge, its KV reservation and its
/// in-flight slot. `settle` closes it with the worker's answer. Dropping it unsettled, as when a
/// handler future is dropped because the client went away, settles it as `Outcome::Failed`.
/// Either way it settles exactly once.
pub struct Ticket {
    // Field order is drop order: the reservation goes before the slot frees, as in Go's Settle.
    hold: Hold,
    _slot: Slot,
}

impl fmt::Debug for Ticket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ticket")
            .field("pod", &self.hold.pod)
            .finish()
    }
}

impl Ticket {
    /// The worker the request was placed on.
    pub fn pod(&self) -> &str {
        &self.hold.pod
    }

    /// Drops the reservation, corrects the tenant's charge to the real usage (or refunds it on
    /// `Failed`), records the run's usage for its next estimate, folds the latency into the
    /// worker's mean service time and releases the slot.
    pub fn settle(mut self, outcome: Outcome) {
        self.hold.outcome = outcome;
    }
}

/// One overflowed request's charge against its tenant. `settle` closes it; dropping it unsettled
/// refunds it.
pub struct OffboxTicket {
    gate: Arc<Shared>,
    charge: Charge,
    outcome: Outcome,
}

impl fmt::Debug for OffboxTicket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OffboxTicket")
            .field("charge", &self.charge)
            .finish()
    }
}

impl OffboxTicket {
    /// Corrects the charge to the real usage, or refunds it on `Failed`, and records the run's
    /// usage for its next estimate.
    pub fn settle(mut self, outcome: Outcome) {
        self.outcome = outcome;
    }
}

impl Drop for OffboxTicket {
    fn drop(&mut self) {
        let mut st = lock(&self.gate.state);
        let now = self.gate.clock.now();
        st.settle_charge(now, &self.charge, &self.outcome);
    }
}

#[derive(Debug)]
struct Reservation {
    id: u64,
    tokens: u64,
    at: Timestamp,
}

/// One worker as the Gate tracks it. `boot` counts restarts, so a run binding made under an
/// older boot is void. `mean_service_s` is an EWMA with alpha 0.2, 0 until the first sample.
#[derive(Debug)]
struct PodState {
    name: String,
    scrape: Option<Scrape>,
    ready: bool,
    boot: u64,
    reservations: Vec<Reservation>,
    mean_service_s: f64,
}

impl PodState {
    /// Sums the reservations the last scrape cannot yet reflect: those made after the scrape
    /// minus `grace`, or all of them before the first scrape.
    fn reserved(&self, grace: Duration) -> u64 {
        self.reservations
            .iter()
            .filter(|r| {
                self.scrape
                    .is_none_or(|s| r.at.saturating_add(grace) > s.at)
            })
            .fold(0, |sum, r| sum.saturating_add(r.tokens))
    }

    fn observe_service(&mut self, latency: Duration) {
        let s = latency.as_secs_f64();
        self.mean_service_s = if self.mean_service_s == 0.0 {
            s
        } else {
            0.2 * s + 0.8 * self.mean_service_s
        };
    }
}

/// One agent run's binding, and the usage vLLM reported for its last settled step.
#[derive(Debug)]
struct Run {
    pod: String,
    boot: u64,
    usage: RunUsage,
    seen: Timestamp,
}

/// A token bucket that refills lazily on every read and never holds more than `burst`. It may
/// go negative, which only the shortfall correction at settle can cause.
#[derive(Debug)]
struct Bucket {
    tokens: f64,
    rate: f64,
    burst: f64,
    last: Timestamp,
}

impl Bucket {
    fn refill(&mut self, now: Timestamp) {
        if now <= self.last {
            return;
        }
        let earned = self.rate * now.saturating_duration_since(self.last).as_secs_f64();
        self.tokens = self.burst.min(self.tokens + earned);
        self.last = now;
    }

    /// A negative `n` charges.
    fn add(&mut self, n: f64) {
        self.tokens = self.burst.min(self.tokens + n);
    }
}

#[derive(Debug)]
struct Buckets {
    named: HashMap<String, Bucket>,
    shared: Bucket,
}

impl Buckets {
    fn of(&mut self, tenant: &str) -> &mut Bucket {
        match self.named.get_mut(tenant) {
            Some(b) => b,
            None => &mut self.shared,
        }
    }
}

struct State {
    pods: Vec<PodState>,
    runs: HashMap<RunId, Run>,
    buckets: Buckets,
    next_reservation: u64,
    rnd: Box<dyn FnMut(usize) -> usize + Send>,
}

impl State {
    fn pod_mut(&mut self, name: &str) -> Option<&mut PodState> {
        self.pods.iter_mut().find(|p| p.name == name)
    }

    /// The worker a run is bound to, unless the run is unknown or its worker restarted since.
    fn bound_pod(&self, run: &RunId) -> Option<&str> {
        let r = self.runs.get(run)?;
        self.pods
            .iter()
            .any(|p| p.name == r.pod && p.boot == r.boot)
            .then_some(r.pod.as_str())
    }

    fn settle_charge(&mut self, now: Timestamp, charge: &Charge, outcome: &Outcome) {
        let b = self.buckets.of(&charge.tenant);
        b.refill(now);
        match outcome {
            Outcome::Failed => b.add(charge.est as f64),
            Outcome::Answered { usage, .. } => {
                let used = usage.prompt_tokens.saturating_add(usage.completion_tokens);
                b.add(charge.est as f64 - used as f64);
                if let Some(run) = charge.run.as_ref()
                    && let Some(r) = self.runs.get_mut(run)
                {
                    r.usage = *usage;
                    r.seen = now;
                }
            }
        }
    }

    /// One `WorkerState` per configured pod. A worker never scraped reports an age of
    /// `down_after`, so `pick` treats it as down.
    fn worker_states(&self, now: Timestamp, cfg: &Config, queue: &Queue) -> Vec<WorkerState> {
        self.pods
            .iter()
            .map(|p| {
                let depth = queue.depth(&p.name);
                let s = p.scrape.as_ref();
                WorkerState {
                    view: WorkerView {
                        pod: p.name.clone(),
                        kv_pool_tokens: s.map_or(0, |s| s.kv_pool_tokens),
                        kv_usage: s.map_or(0.0, |s| s.kv_usage),
                        reserved_tokens: p.reserved(cfg.reserve_grace),
                        in_flight: depth.in_flight,
                        queued: depth.queued(),
                        max_inflight: cfg.max_inflight,
                        mean_service_s: p.mean_service_s,
                        ttft_p50_s: s.map_or(0.0, |s| s.ttft_p50_s),
                        ttft_p99_s: s.map_or(0.0, |s| s.ttft_p99_s),
                        waiting_rising: s.is_some_and(|s| s.waiting_rising),
                    },
                    ready: p.ready && s.is_some(),
                    age: s.map_or(cfg.staleness.down_after, |s| {
                        now.saturating_duration_since(s.at)
                    }),
                }
            })
            .collect()
    }
}

struct Shared {
    cfg: Config,
    clock: Clock,
    queue: Queue,
    state: Mutex<State>,
}

/// The gateway's stateful core: the worker views, the reservations, the run table and the
/// tenant buckets, all under one lock, and the `Queue` behind them. Deciding and reserving
/// happen under that one lock, so two requests can never be placed on the same free KV. The
/// lock may be held while reading the queue's depth and is never held across an `.await`.
/// Cloning is cheap and every clone is the same Gate.
#[derive(Clone)]
pub struct Gate(Arc<Shared>);

impl fmt::Debug for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gate").field("cfg", &self.0.cfg).finish()
    }
}

/// A placed decision whose charge and reservation are made, waiting for a slot.
struct Pending {
    decision: Decision,
    hold: Hold,
    start: Timestamp,
}

impl Gate {
    /// `rnd(n)` returns a value in `[0, n)` for `PickPolicy::P2c`. Every configured tenant starts
    /// with a full bucket.
    pub fn new(
        cfg: Config,
        clock: Clock,
        rnd: impl FnMut(usize) -> usize + Send + 'static,
    ) -> Self {
        let start = clock.now();
        let bucket = |lim: &TenantLimit| Bucket {
            tokens: lim.burst,
            rate: lim.rate_per_s,
            burst: lim.burst,
            last: start,
        };
        let shared = cfg.tenants.get(SHARED_TENANT).map_or(
            Bucket {
                tokens: 0.0,
                rate: 0.0,
                burst: 0.0,
                last: start,
            },
            bucket,
        );
        let named = cfg
            .tenants
            .iter()
            .filter(|(name, _)| *name != SHARED_TENANT)
            .map(|(name, lim)| (name.clone(), bucket(lim)))
            .collect();
        let pods = cfg
            .pods
            .iter()
            .map(|name| PodState {
                name: name.clone(),
                scrape: None,
                ready: false,
                boot: 0,
                reservations: Vec::new(),
                mean_service_s: 0.0,
            })
            .collect();
        let state = State {
            pods,
            runs: HashMap::new(),
            buckets: Buckets { named, shared },
            next_reservation: 0,
            rnd: Box::new(rnd),
        };
        Self(Arc::new(Shared {
            queue: Queue::new(&cfg.pods, cfg.max_inflight, cfg.max_queued, clock.clone()),
            cfg,
            clock,
            state: Mutex::new(state),
        }))
    }

    /// Records a worker's scrape and prunes runs idle past `run_ttl`. An unknown pod is ignored.
    pub fn observe(&self, pod: &str, scrape: Scrape) {
        let g = &*self.0;
        let mut st = lock(&g.state);
        let Some(p) = st.pod_mut(pod) else {
            return;
        };
        p.scrape = Some(scrape);
        let now = g.clock.now();
        st.runs
            .retain(|_, r| r.seen.saturating_add(g.cfg.run_ttl) >= now);
    }

    /// Records the warm-up gate's verdict. An unknown pod is ignored.
    pub fn set_ready(&self, pod: &str, ready: bool) {
        if let Some(p) = lock(&self.0.state).pod_mut(pod) {
            p.ready = ready;
        }
    }

    /// Records that a worker came back from a Down period. Its boot advances, which unbinds every
    /// run bound under the old boot; its reservations are cleared because the requests behind
    /// them are gone; and it stays not ready until warm-up says otherwise.
    pub fn restarted(&self, pod: &str) {
        if let Some(p) = lock(&self.0.state).pod_mut(pod) {
            p.boot = p.boot.saturating_add(1);
            p.ready = false;
            p.reservations.clear();
        }
    }

    /// Decides, reserves and queues one request. `pick` runs under the lock with the run's
    /// binding and the prompt length of its last settled step. When a worker is chosen, the
    /// tenant is charged, the reservation recorded and the run bound before the lock is released
    /// to wait for a slot. A refusal from `pick` charges nothing. A refusal from the queue undoes
    /// the charge and the reservation and becomes a 503 placement with no pod; the run binding
    /// outlives it, because the run's history is wherever its last settled step ran. Dropping
    /// the future while it waits undoes the same.
    pub async fn admit(&self, r: &Request) -> Admission {
        let Pending {
            mut decision,
            hold,
            start,
        } = match self.decide(r) {
            Ok(pending) => pending,
            Err(refused) => return Admission::Refused(refused),
        };
        match self
            .0
            .queue
            .acquire(&hold.pod, r.priority, r.deadline)
            .await
        {
            Ok(slot) => {
                decision.queued = self.0.clock.now().saturating_duration_since(start);
                Admission::Admitted(decision, Ticket { hold, _slot: slot })
            }
            Err(refusal) => {
                drop(hold);
                decision.placement.result = Err(Refusal {
                    reason: refusal.reason(),
                    retry_after: Duration::from_secs(1),
                    detail: String::new(),
                });
                Admission::Refused(decision)
            }
        }
    }

    fn decide(&self, r: &Request) -> Result<Pending, Decision> {
        let g = &*self.0;
        let mut st = lock(&g.state);
        let start = g.clock.now();

        let prior = r
            .run
            .as_ref()
            .and_then(|run| st.runs.get(run))
            .map(|run| run.usage);
        let bound = r
            .run
            .as_ref()
            .and_then(|run| st.bound_pod(run))
            .map(str::to_owned);
        let est = estimate_tokens(r, prior.as_ref());
        let history = prior.map_or(0, |u| u.prompt_tokens);

        let bucket = st.buckets.of(&r.tenant);
        bucket.refill(start);
        let tenant = TenantState {
            available_tokens: bucket.tokens,
            rate_per_s: bucket.rate,
        };
        let fleet = Fleet {
            now: start,
            workers: st.worker_states(start, &g.cfg, &g.queue),
            tenant,
            est_tokens: est,
            bound: bound.clone(),
            history,
            shed: g.cfg.shed,
            staleness: g.cfg.staleness,
            stick_slack: g.cfg.stick_slack,
        };
        let placement = pick(r, &fleet, g.cfg.policy, &mut st.rnd);
        let mut decision = Decision {
            placement,
            est,
            queued: Duration::ZERO,
            from: bound,
            history,
        };
        let Some(pod) = decision.placement.pod().map(str::to_owned) else {
            return Err(decision);
        };

        st.next_reservation = st.next_reservation.wrapping_add(1);
        let id = st.next_reservation;
        let reserved = st.pod_mut(&pod).map(|p| {
            p.reservations.push(Reservation {
                id,
                tokens: est,
                at: start,
            });
            p.boot
        });
        let Some(boot) = reserved else {
            // `pick` chooses only among the pods it was given, so this cannot happen.
            decision.placement.result = Err(Refusal {
                reason: Reason::NoEligiblePod,
                retry_after: Duration::from_secs(2),
                detail: format!("picked unknown pod {pod}"),
            });
            return Err(decision);
        };
        st.buckets.of(&r.tenant).add(-(est as f64));
        if let Some(run) = &r.run {
            st.runs.insert(
                run.clone(),
                Run {
                    pod: pod.clone(),
                    boot,
                    usage: prior.unwrap_or_default(),
                    seen: start,
                },
            );
        }
        let hold = Hold {
            gate: Arc::clone(&self.0),
            charge: Charge {
                tenant: r.tenant.clone(),
                run: r.run.clone(),
                est,
            },
            pod,
            reservation: id,
            outcome: Outcome::Failed,
        };
        Ok(Pending {
            decision,
            hold,
            start,
        })
    }

    /// A tenant's bucket after refilling it to now. An unknown tenant reads the shared bucket.
    pub fn tenant_tokens(&self, tenant: &str) -> f64 {
        let mut st = lock(&self.0.state);
        let b = st.buckets.of(tenant);
        b.refill(self.0.clock.now());
        b.tokens
    }

    /// Overwrites a tenant's bucket. For tests in this and other crates only.
    #[doc(hidden)]
    pub fn set_tenant_tokens_for_test(&self, tenant: &str, tokens: f64) {
        let mut st = lock(&self.0.state);
        let b = st.buckets.of(tenant);
        b.refill(self.0.clock.now());
        b.tokens = tokens;
    }

    /// Charges a tenant for a request that is about to leave the box, with the estimate `admit`
    /// would have used, or returns `None` without charging when the bucket cannot cover it. An
    /// overflowed request is served and paid for by someone, so it spends quota exactly like a
    /// local answer; without this a tenant could spend the backend's tokens without limit
    /// whenever the box is full.
    pub fn charge_offbox(&self, r: &Request) -> Option<OffboxTicket> {
        let g = &*self.0;
        let mut st = lock(&g.state);
        let prior = r
            .run
            .as_ref()
            .and_then(|run| st.runs.get(run))
            .map(|run| run.usage);
        let est = estimate_tokens(r, prior.as_ref());
        let b = st.buckets.of(&r.tenant);
        b.refill(g.clock.now());
        if est as f64 > b.tokens {
            return None;
        }
        b.add(-(est as f64));
        Some(OffboxTicket {
            gate: Arc::clone(&self.0),
            charge: Charge {
                tenant: r.tenant.clone(),
                run: r.run.clone(),
                est,
            },
            outcome: Outcome::Failed,
        })
    }

    /// The per-worker states `admit` would build right now, in `Config::pods` order.
    pub fn view(&self) -> Vec<WorkerState> {
        let g = &*self.0;
        let st = lock(&g.state);
        st.worker_states(g.clock.now(), &g.cfg, &g.queue)
    }

    /// The worker a run is bound to, or `None` when the run is unknown or its worker restarted
    /// since the binding.
    pub fn bound(&self, run: &RunId) -> Option<String> {
        lock(&self.0.state).bound_pod(run).map(str::to_owned)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use ariadne_decide::{DataClass, Priority, Sticky, parse_request_id};

    use super::*;
    use crate::clock::ManualClock;

    pub(in crate::fleet) fn zero_rnd(_: usize) -> usize {
        0
    }

    fn new_test_gate(cfg: Config) -> (Gate, Clock, ManualClock) {
        let (clock, hand) = Clock::manual();
        (Gate::new(cfg, clock.clone(), zero_rnd), clock, hand)
    }

    fn pods(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_owned()).collect()
    }

    fn scrape_ready(g: &Gate, clock: &Clock, pod: &str, pool: u64, usage: f64) {
        g.observe(pod, scrape(clock.now(), pool, usage));
        g.set_ready(pod, true);
    }

    fn scrape(at: Timestamp, pool: u64, usage: f64) -> Scrape {
        Scrape {
            at,
            kv_pool_tokens: pool,
            kv_usage: usage,
            ttft_p50_s: 0.0,
            ttft_p99_s: 0.0,
            waiting_rising: false,
        }
    }

    /// An interactive request. The run comes from the id's `-s<n>` suffix, as `inspect` parses it.
    pub(in crate::fleet) fn request(
        now: Timestamp,
        id: &str,
        tenant: &str,
        body_bytes: usize,
        max_out: u64,
    ) -> Request {
        let (run, step) = parse_request_id(id).map_or((None, 0), |(run, step)| (Some(run), step));
        Request {
            id: id.to_owned(),
            run,
            step,
            tenant: tenant.to_owned(),
            app: String::new(),
            priority: Priority::Interactive,
            class: DataClass::Restricted,
            body_bytes,
            max_out,
            arrived: now,
            deadline: now.saturating_add(Duration::from_secs(10)),
        }
    }

    fn run(name: &str) -> RunId {
        RunId::new(name).expect("run id")
    }

    fn answered(prompt: u64, completion: u64, body_bytes: usize, latency_ms: u64) -> Outcome {
        Outcome::Answered {
            usage: RunUsage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                body_bytes,
            },
            latency: Duration::from_millis(latency_ms),
        }
    }

    pub(in crate::fleet) async fn must_admit(g: &Gate, r: &Request) -> (Decision, Ticket) {
        match g.admit(r).await {
            Admission::Admitted(d, t) => (d, t),
            Admission::Refused(d) => panic!("admit({}) refused: {:?}", r.id, d.placement),
        }
    }

    async fn must_refuse(g: &Gate, r: &Request, code: u16, reason: Reason) -> Decision {
        let d = match g.admit(r).await {
            Admission::Refused(d) => d,
            Admission::Admitted(_, t) => panic!("admit({}) placed on {}", r.id, t.pod()),
        };
        let refusal = d.placement.result.as_ref().expect_err("refused");
        assert_eq!(
            (refusal.code(), refusal.reason),
            (code, reason),
            "{refusal:?}"
        );
        assert_eq!(d.placement.pod(), None);
        d
    }

    fn detail(d: &Decision) -> &str {
        &d.placement.result.as_ref().expect_err("refused").detail
    }

    pub(in crate::fleet) fn view_of(g: &Gate, pod: &str) -> WorkerView {
        g.view()
            .into_iter()
            .find(|w| w.view.pod == pod)
            .unwrap_or_else(|| panic!("view has no pod {pod}"))
            .view
    }

    fn assert_reserved(g: &Gate, pod: &str, want: u64) {
        assert_eq!(
            view_of(g, pod).reserved_tokens,
            want,
            "{pod} reserved_tokens"
        );
    }

    fn assert_placed(d: &Decision, t: &Ticket, pod: &str, sticky: Sticky, est: u64) {
        assert_eq!((d.placement.pod(), t.pod()), (Some(pod), pod), "placed on");
        assert_eq!(d.placement.sticky, sticky);
        assert_eq!(d.est, est);
    }

    #[test]
    fn admit_future_is_send() {
        fn send<T: Send>(_: T) {}
        let (g, clock, _) = new_test_gate(Config::defaults(pods(&["a"])));
        let r = request(clock.now(), "req-1", "platform", 1, 1);
        send(g.admit(&r));
    }

    #[tokio::test]
    async fn new_run_lands_on_less_loaded_pod_and_sticks() {
        let (g, clock, hand) = new_test_gate(Config::defaults(pods(&["a", "b"])));
        scrape_ready(&g, &clock, "a", 100_000, 0.5);
        scrape_ready(&g, &clock, "b", 100_000, 0.2);

        let (d1, t1) =
            must_admit(&g, &request(clock.now(), "run1-s1", "platform", 7000, 768)).await;
        assert_placed(&d1, &t1, "b", Sticky::New, 2768);
        assert_eq!(g.bound(&run("run1")).as_deref(), Some("b"));
        t1.settle(answered(9000, 60, 7000, 300));

        hand.advance(Duration::from_secs(1));
        scrape_ready(&g, &clock, "a", 100_000, 0.5);
        scrape_ready(&g, &clock, "b", 100_000, 0.6);

        let (d2, t2) =
            must_admit(&g, &request(clock.now(), "run1-s2", "platform", 10500, 768)).await;
        assert_placed(&d2, &t2, "b", Sticky::Hit, 10828);
        assert_eq!(view_of(&g, "b").mean_service_s, 0.3);
        t2.settle(answered(10000, 50, 10500, 100));
        assert_eq!(view_of(&g, "b").mean_service_s, 0.2 * 0.1 + 0.8 * 0.3);
    }

    #[tokio::test]
    async fn decision_names_where_a_moved_run_left_its_history() {
        let (g, clock, hand) = new_test_gate(Config::defaults(pods(&["a", "b"])));
        scrape_ready(&g, &clock, "a", 100_000, 0.5);
        scrape_ready(&g, &clock, "b", 100_000, 0.2);

        let (d1, t1) =
            must_admit(&g, &request(clock.now(), "run1-s1", "platform", 7000, 768)).await;
        assert_eq!((d1.from.as_deref(), d1.history), (None, 0), "step 1");
        t1.settle(answered(9000, 60, 7000, 1000));

        // b, the bound worker, is now 0.5 load units busier than a, over the 0.25 slack, but
        // still above the KV line, so the run moves for load.
        hand.advance(Duration::from_secs(1));
        scrape_ready(&g, &clock, "a", 100_000, 0.2);
        scrape_ready(&g, &clock, "b", 100_000, 0.7);
        let (d2, _t2) =
            must_admit(&g, &request(clock.now(), "run1-s2", "platform", 10500, 768)).await;
        assert_eq!(
            (d2.placement.pod(), d2.placement.sticky),
            (Some("a"), Sticky::BrokenLoad)
        );
        assert_eq!(
            (d2.from.as_deref(), d2.history),
            (Some("b"), 9000),
            "step 2"
        );
        assert_eq!(g.bound(&run("run1")).as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn unusable_pods_are_never_picked() {
        let (g, clock, _) = new_test_gate(Config::defaults(pods(&["a", "b", "c"])));
        g.set_ready("a", true);
        g.observe("b", scrape(clock.now(), 100_000, 0.0));

        let r = request(clock.now(), "req-1", "platform", 7000, 768);
        let d = must_refuse(&g, &r, 503, Reason::NoEligiblePod).await;
        assert_eq!(d.est, 2768);

        scrape_ready(&g, &clock, "c", 100_000, 0.7);
        let (d, t) = must_admit(&g, &request(clock.now(), "req-2", "platform", 7000, 768)).await;
        assert_placed(&d, &t, "c", Sticky::None, 2768);
        t.settle(Outcome::Failed);
    }

    #[tokio::test]
    async fn reservations_count_until_settled_or_superseded() {
        let (g, clock, _) = new_test_gate(Config::defaults(pods(&["a", "b"])));
        scrape_ready(&g, &clock, "a", 10000, 0.0);
        scrape_ready(&g, &clock, "b", 10000, 0.0);
        let scraped_at = clock.now();

        let (d1, t1) = must_admit(&g, &request(clock.now(), "req-1", "platform", 3500, 7000)).await;
        assert_placed(&d1, &t1, "a", Sticky::None, 8000);
        assert_reserved(&g, "a", 8000);
        assert_reserved(&g, "b", 0);

        let (d2, t2) = must_admit(&g, &request(clock.now(), "req-2", "platform", 3500, 7000)).await;
        assert_placed(&d2, &t2, "b", Sticky::None, 8000);
        assert_reserved(&g, "a", 8000);
        assert_reserved(&g, "b", 8000);

        let r3 = request(clock.now(), "req-3", "platform", 3500, 7000);
        must_refuse(&g, &r3, 503, Reason::KvFree).await;

        t1.settle(answered(1000, 10, 3500, 0));
        assert_reserved(&g, "a", 0);
        assert_reserved(&g, "b", 8000);

        let at = |ms| scraped_at.saturating_add(Duration::from_millis(ms));
        g.observe("b", scrape(at(500), 10000, 0.0));
        assert_reserved(&g, "b", 8000);

        g.observe("b", scrape(at(5000), 10000, 0.8));
        assert_reserved(&g, "b", 0);
        t2.settle(answered(1000, 10, 3500, 0));
        assert_reserved(&g, "b", 0);
    }

    #[tokio::test]
    async fn tenant_bucket_charges_and_refunds() {
        let mut cfg = Config::defaults(pods(&["a"]));
        cfg.tenants = HashMap::from([
            (
                "platform".to_owned(),
                TenantLimit {
                    rate_per_s: 100.0,
                    burst: 5000.0,
                },
            ),
            (
                "*".to_owned(),
                TenantLimit {
                    rate_per_s: 10.0,
                    burst: 3000.0,
                },
            ),
        ]);
        let (g, clock, _) = new_test_gate(cfg);
        scrape_ready(&g, &clock, "a", 1_000_000, 0.0);
        let req =
            |id: &str, tenant: &str, body, max_out| request(clock.now(), id, tenant, body, max_out);

        let d = must_refuse(
            &g,
            &req("big", "platform", 3500, 7000),
            429,
            Reason::TenantTokens,
        )
        .await;
        assert_eq!(
            detail(&d),
            "need 8000 tokens, tenant has 5000 available at 100/s"
        );
        assert_eq!(
            d.placement.result.as_ref().map_err(|r| r.retry_after).err(),
            Some(Duration::from_secs(30))
        );

        let (da, ta) = must_admit(&g, &req("A", "platform", 7000, 2000)).await;
        assert_placed(&da, &ta, "a", Sticky::None, 4000);

        let d = must_refuse(
            &g,
            &req("B", "platform", 3500, 1500),
            429,
            Reason::TenantTokens,
        )
        .await;
        assert_eq!(
            detail(&d),
            "need 2500 tokens, tenant has 1000 available at 100/s"
        );

        ta.settle(answered(2000, 500, 7000, 0));
        let (db, tb) = must_admit(&g, &req("B", "platform", 3500, 1500)).await;
        assert_placed(&db, &tb, "a", Sticky::None, 2500);
        tb.settle(answered(1000, 10, 3500, 0));

        let (dx, tx) = must_admit(&g, &req("X", "tenant-x", 3500, 1500)).await;
        assert_placed(&dx, &tx, "a", Sticky::None, 2500);
        let d = must_refuse(
            &g,
            &req("Y", "tenant-y", 3500, 1500),
            429,
            Reason::TenantTokens,
        )
        .await;
        assert_eq!(
            detail(&d),
            "need 2500 tokens, tenant has 500 available at 10/s"
        );
        tx.settle(Outcome::Failed);
    }

    #[tokio::test]
    async fn restart_breaks_binding_and_clears_reservations() {
        let (g, clock, _) = new_test_gate(Config::defaults(pods(&["a", "b"])));
        scrape_ready(&g, &clock, "a", 100_000, 0.2);
        scrape_ready(&g, &clock, "b", 100_000, 0.5);

        let (d1, t1) =
            must_admit(&g, &request(clock.now(), "run1-s1", "platform", 7000, 768)).await;
        assert_placed(&d1, &t1, "a", Sticky::New, 2768);
        assert_reserved(&g, "a", 2768);
        assert_eq!(g.bound(&run("run1")).as_deref(), Some("a"));

        g.restarted("a");
        assert_eq!(g.bound(&run("run1")), None, "after restart");
        assert_reserved(&g, "a", 0);

        let (d2, t2) =
            must_admit(&g, &request(clock.now(), "run1-s2", "platform", 7000, 768)).await;
        assert_placed(&d2, &t2, "b", Sticky::New, 2768);
        assert_eq!(g.bound(&run("run1")).as_deref(), Some("b"), "after rebind");

        t1.settle(Outcome::Failed);
        t2.settle(Outcome::Failed);
        assert_reserved(&g, "a", 0);
        assert_reserved(&g, "b", 0);
    }

    #[tokio::test]
    async fn second_admit_waits_for_settle() {
        let mut cfg = Config::defaults(pods(&["a"]));
        cfg.max_inflight = 1;
        let (g, clock, _) = new_test_gate(cfg);
        scrape_ready(&g, &clock, "a", 1_000_000, 0.0);

        let (d1, t1) = must_admit(&g, &request(clock.now(), "req-1", "platform", 7000, 768)).await;
        assert_placed(&d1, &t1, "a", Sticky::None, 2768);

        let second = tokio::spawn({
            let (g, r) = (
                g.clone(),
                request(clock.now(), "req-2", "platform", 7000, 768),
            );
            async move { g.admit(&r).await }
        });
        for _ in 0..10_000 {
            if view_of(&g, "a").queued == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(!second.is_finished(), "second admit returned before settle");
        let v = view_of(&g, "a");
        assert_eq!((v.in_flight, v.queued), (1, 1));

        t1.settle(answered(2000, 100, 7000, 0));
        let admission = tokio::time::timeout(Duration::from_secs(2), second)
            .await
            .expect("second admit did not return after settle")
            .expect("admit task");
        let Admission::Admitted(d2, t2) = admission else {
            panic!("second admit refused: {admission:?}");
        };
        assert_placed(&d2, &t2, "a", Sticky::None, 2768);
        let v = view_of(&g, "a");
        assert_eq!((v.in_flight, v.queued), (1, 0));

        t2.settle(Outcome::Failed);
        let v = view_of(&g, "a");
        assert_eq!((v.in_flight, v.reserved_tokens), (0, 0));
    }

    #[tokio::test]
    async fn dropping_a_waiting_admit_undoes_its_charge_and_reservation() {
        let mut cfg = Config::defaults(pods(&["a"]));
        cfg.max_inflight = 1;
        let (g, clock, _) = new_test_gate(cfg);
        scrape_ready(&g, &clock, "a", 1_000_000, 0.0);
        let full = g.tenant_tokens("platform");

        let (_, t1) = must_admit(&g, &request(clock.now(), "req-1", "platform", 7000, 768)).await;
        let second = tokio::spawn({
            let (g, r) = (
                g.clone(),
                request(clock.now(), "req-2", "platform", 7000, 768),
            );
            async move { g.admit(&r).await }
        });
        for _ in 0..10_000 {
            if view_of(&g, "a").queued == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(view_of(&g, "a").reserved_tokens, 2 * 2768);
        second.abort();
        assert!(second.await.is_err_and(|e| e.is_cancelled()));
        let v = view_of(&g, "a");
        assert_eq!((v.in_flight, v.queued, v.reserved_tokens), (1, 0, 2768));
        assert_eq!(g.tenant_tokens("platform"), full - 2768.0);

        drop(t1);
        let v = view_of(&g, "a");
        assert_eq!(
            (v.in_flight, v.reserved_tokens),
            (0, 0),
            "an unsettled ticket settles on drop"
        );
        assert_eq!(g.tenant_tokens("platform"), full);
    }

    #[tokio::test]
    async fn settled_ticket_charges_real_usage_once() {
        let mut cfg = Config::defaults(pods(&["a"]));
        cfg.tenants = HashMap::from([(
            "platform".to_owned(),
            TenantLimit {
                rate_per_s: 100.0,
                burst: 5000.0,
            },
        )]);
        let (g, clock, _) = new_test_gate(cfg);
        scrape_ready(&g, &clock, "a", 1_000_000, 0.0);

        let (da, ta) = must_admit(&g, &request(clock.now(), "A", "platform", 7000, 2000)).await;
        assert_placed(&da, &ta, "a", Sticky::None, 4000);
        ta.settle(answered(2000, 1000, 7000, 0));

        let v = view_of(&g, "a");
        assert_eq!((v.in_flight, v.reserved_tokens), (0, 0));
        let r = request(clock.now(), "B", "platform", 3500, 1500);
        let d = must_refuse(&g, &r, 429, Reason::TenantTokens).await;
        assert_eq!(
            detail(&d),
            "need 2500 tokens, tenant has 2000 available at 100/s"
        );
    }

    #[tokio::test]
    async fn offbox_charge_settles_and_refunds() {
        let mut cfg = Config::defaults(pods(&["a"]));
        cfg.tenants = HashMap::from([(
            "platform".to_owned(),
            TenantLimit {
                rate_per_s: 0.0,
                burst: 5000.0,
            },
        )]);
        let (g, clock, _) = new_test_gate(cfg);
        let r = request(clock.now(), "run1-s1", "platform", 3500, 1000);

        let t = g.charge_offbox(&r).expect("covered");
        assert_eq!(g.tenant_tokens("platform"), 3000.0);
        drop(t);
        assert_eq!(
            g.tenant_tokens("platform"),
            5000.0,
            "dropped unsettled refunds"
        );

        g.charge_offbox(&r)
            .expect("covered")
            .settle(answered(1500, 100, 3500, 0));
        assert_eq!(g.tenant_tokens("platform"), 3400.0);

        let big = request(clock.now(), "big", "platform", 35000, 1000);
        assert!(g.charge_offbox(&big).is_none(), "11000 > 3400");
        assert_eq!(
            g.tenant_tokens("platform"),
            3400.0,
            "a refusal charges nothing"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_admit_settle_leaves_nothing_held() {
        let mut cfg = Config::defaults(pods(&["a", "b"]));
        cfg.tenants = HashMap::from([(
            "platform".to_owned(),
            TenantLimit {
                rate_per_s: 1e6,
                burst: 1e6,
            },
        )]);
        let (g, clock, _) = new_test_gate(cfg);
        scrape_ready(&g, &clock, "a", 1_000_000, 0.0);
        scrape_ready(&g, &clock, "b", 1_000_000, 0.0);

        let tasks: Vec<_> = (0..50)
            .map(|i| {
                let g = g.clone();
                let r = request(clock.now(), &format!("run{i}-s1"), "platform", 3500, 768);
                tokio::spawn(async move {
                    match g.admit(&r).await {
                        Admission::Admitted(_, t) => {
                            t.settle(answered(1000, 100, 3500, 50));
                            true
                        }
                        Admission::Refused(_) => false,
                    }
                })
            })
            .collect();
        let mut placed = 0;
        for t in tasks {
            placed += usize::from(t.await.expect("admit task"));
        }
        assert_eq!(placed, 50);
        for pod in ["a", "b"] {
            let v = view_of(&g, pod);
            assert_eq!(
                (v.reserved_tokens, v.in_flight, v.queued),
                (0, 0, 0),
                "{pod}"
            );
        }
    }
}
