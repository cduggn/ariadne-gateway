//! The gateway's Prometheus metrics. `observe` feeds the request counters and the latency
//! histogram from each `serve::Event`, and `render` derives every replica gauge from the Gate's
//! view and the Fleet's status at scrape time, so a gauge can never drift from the state the
//! gateway routes on. The text is what client_golang v1.24 wrote for the Go gateway, without the
//! `go_*` and `process_*` families.

mod text;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use ariadne_decide::{DEFAULT_POLICY, DataClass, Reason as ShedReason, WorkerState};

use self::text::{Exposition, Histogram, Kind, Spec};
use crate::fleet::{Fleet, Gate, Phase, WorkerStatus};
use crate::hop;
use crate::overflow::{self, Overflow};
use crate::serve::{Event, Reason};

const fn counter<const N: usize>(
    name: &'static str,
    help: &'static str,
    labels: [&'static str; N],
) -> Spec<N> {
    Spec {
        name,
        help,
        kind: Kind::Counter,
        labels,
    }
}

const fn gauge<const N: usize>(
    name: &'static str,
    help: &'static str,
    labels: [&'static str; N],
) -> Spec<N> {
    Spec {
        name,
        help,
        kind: Kind::Gauge,
        labels,
    }
}

const REQUESTS: Spec<2> = counter(
    "orch_requests_total",
    "Requests received, including those the guard rejected.",
    ["priority", "class"],
);
const SHED: Spec<2> = counter(
    "orch_shed_total",
    "Requests refused by the gateway, by reason and status code.",
    ["reason", "code"],
);
const PICK: Spec<2> = counter(
    "orch_pick_total",
    "Requests placed on a worker, by pod and pick policy.",
    ["pod", "policy"],
);
const PICK_UNKNOWN: Spec<1> = counter(
    "orch_pick_unknown_snapshot_total",
    "Placements made on a worker whose telemetry was stale.",
    ["pod"],
);
const STICKY: Spec<1> = counter(
    "orch_sticky_total",
    "Stickiness outcomes relative to the run's bound worker.",
    ["outcome"],
);
const COMPLETED: Spec<3> = counter(
    "orch_completed_total",
    "Worker answers relayed to the client, by pod, status and finish reason.",
    ["pod", "status", "finish_reason"],
);
const UPSTREAM_ERRORS: Spec<2> = counter(
    "orch_upstream_errors_total",
    "Upstream failures, transport for no answer and status for a non-200 answer.",
    ["pod", "kind"],
);
const PROMPT_TOKENS: Spec<2> = counter(
    "orch_prompt_tokens_total",
    "Prompt tokens by cache outcome: shared_hit, run_hit or miss.",
    ["pod", "kind"],
);
const COMPLETION_TOKENS: Spec<1> = counter(
    "orch_completion_tokens_total",
    "Completion tokens generated per pod.",
    ["pod"],
);
const OVERFLOW: Spec<1> = counter(
    "orch_overflow_total",
    "Overflow decisions for gateway 503s: blocked_invariant kept a restricted request on the box, no_backend had nowhere to send it, sent relayed the backend's answer; failed, unavailable, capped and quota answered the 503.",
    ["result"],
);
const OVERFLOW_AVAILABLE: Spec<0> = gauge(
    "orch_overflow_backend_available",
    "1 while the overflow backend's last probe passed and no send has failed since; 0 without a backend.",
    [],
);
const HOP: Spec<1> = counter(
    "orch_hop_total",
    "KV hops for runs moved off a worker that still held their history: hopped, failed or busy (the worker recomputed), below_threshold or recompute_cheaper.",
    ["result"],
);
const RESTRICTED_OFFBOX: Spec<0> = counter(
    "orch_restricted_offbox_total",
    "Restricted requests routed off the box. Must stay 0.",
    [],
);
const DURATION: Spec<1> = Spec {
    name: "orch_request_duration_seconds",
    help: "Request time by stage: gateway, queue, hop (KV copy from the run's old worker), local (upstream), overflow (the off-box backend) and e2e.",
    kind: Kind::Histogram,
    labels: ["stage"],
};

const REPLICA_PHASE: Spec<3> = gauge(
    "orch_replica_phase",
    "1 for the worker's current phase, 0 for the others.",
    ["pool", "pod", "phase"],
);
const REPLICA_RESTARTS: Spec<2> = counter(
    "orch_replica_restarts_total",
    "Down to Warming transitions, including the first at startup.",
    ["pool", "pod"],
);
const KV_FREE: Spec<1> = gauge(
    "orch_kv_free_ratio",
    "Free KV over the Ready workers as a fraction of their pools, 0 when none is Ready.",
    ["pool"],
);

/// A worker's free KV ratio below this is saturating: the admission line.
const SATURATED_BELOW: f64 = 1.0 - DEFAULT_POLICY.kv_line;

type Read = fn(&WorkerState, &WorkerStatus) -> f64;

/// Every `{pool, pod}` gauge and how to read it from the Gate's view of the worker and the
/// Fleet's status for it.
const REPLICA_GAUGES: [(Spec<2>, Read); 10] = [
    (
        gauge(
            "orch_replica_healthy",
            "1 while the gate may place on the worker.",
            ["pool", "pod"],
        ),
        |ws, _| b2f(ws.ready),
    ),
    (
        gauge(
            "orch_replica_warm",
            "1 while the fleet's warm-up gate holds the worker Ready.",
            ["pool", "pod"],
        ),
        |_, st| b2f(st.phase == Phase::Ready),
    ),
    (
        gauge(
            "orch_replica_saturating",
            "1 while the worker's free KV is below the admission line.",
            ["pool", "pod"],
        ),
        |ws, _| b2f(ws.view.free_ratio() < SATURATED_BELOW),
    ),
    (
        gauge(
            "orch_replica_kv_free_ratio",
            "Free KV as a fraction of the pool, net of the gateway's reservations.",
            ["pool", "pod"],
        ),
        |ws, _| ws.view.free_ratio(),
    ),
    (
        gauge(
            "orch_replica_tokens_in_flight",
            "Tokens the gateway has reserved on the worker since its last scrape.",
            ["pool", "pod"],
        ),
        |ws, _| ws.view.reserved_tokens as f64,
    ),
    (
        gauge(
            "orch_replica_active_requests",
            "Requests the gateway holds in flight on the worker.",
            ["pool", "pod"],
        ),
        |ws, _| ws.view.in_flight as f64,
    ),
    (
        gauge(
            "orch_replica_queue_depth",
            "Requests waiting in the gateway's queue for the worker.",
            ["pool", "pod"],
        ),
        |ws, _| ws.view.queued as f64,
    ),
    (
        gauge(
            "orch_replica_snapshot_age_seconds",
            "Seconds since the worker's last successful scrape.",
            ["pool", "pod"],
        ),
        |ws, _| ws.age.as_secs_f64(),
    ),
    (
        gauge(
            "orch_replica_mean_service_seconds",
            "Recent mean seconds per request on the worker, 0 until the first.",
            ["pool", "pod"],
        ),
        |ws, _| ws.view.mean_service_s,
    ),
    (
        gauge(
            "orch_warmup_probe_seconds",
            "Latency of the worker's latest warm-up probe, the re-quoted TTFT.",
            ["pool", "pod"],
        ),
        |_, st| st.last_probe_s,
    ),
];

const PHASES: [Phase; 3] = [Phase::Down, Phase::Warming, Phase::Ready];

/// The shed pairs exposed at 0 from startup, so a rate() over a reason that has never fired
/// reads 0 instead of no data.
const PREREGISTERED_SHEDS: [ShedReason; 6] = [
    ShedReason::TenantTokens,
    ShedReason::KvFree,
    ShedReason::TimeoutQueue,
    ShedReason::P99Spread,
    ShedReason::NoEligiblePod,
    ShedReason::QueueFull,
];

/// The pod label of an answer the overflow backend gave, so `orch_completed_total` tells
/// served-off-box from served-by-a-worker.
const OVERFLOW_POD: &str = "overflow";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Gateway,
    Queue,
    Hop,
    Local,
    Overflow,
    E2e,
}

impl Stage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Gateway => "gateway",
            Self::Queue => "queue",
            Self::Hop => "hop",
            Self::Local => "local",
            Self::Overflow => "overflow",
            Self::E2e => "e2e",
        }
    }
}

/// One counter family's series, keyed by label values in its `Spec`'s label order.
type Series<const N: usize> = HashMap<[String; N], u64>;

fn add<const N: usize>(series: &mut Series<N>, values: [&str; N], n: u64) {
    *series.entry(values.map(str::to_owned)).or_default() += n;
}

fn render_counters<const N: usize>(out: &mut Exposition, spec: &Spec<N>, series: &Series<N>) {
    for (values, n) in series {
        out.sample(spec, values.each_ref().map(String::as_str), *n as f64);
    }
}

/// Everything `observe` has counted. Every increment is whole, so the counters are integers.
#[derive(Debug, Default)]
struct Counters {
    requests: Series<2>,
    shed: Series<2>,
    pick: Series<2>,
    pick_unknown: Series<1>,
    sticky: Series<1>,
    completed: Series<3>,
    upstream_errors: Series<2>,
    prompt_tokens: Series<2>,
    completion_tokens: Series<1>,
    overflow: Series<1>,
    hop: Series<1>,
    restricted_offbox: Series<0>,
    duration: BTreeMap<Stage, Histogram>,
}

impl Counters {
    fn preregistered() -> Self {
        let mut c = Self::default();
        for r in PREREGISTERED_SHEDS {
            add(&mut c.shed, [r.as_str(), &r.code().to_string()], 0);
        }
        for o in overflow::Outcome::ALL {
            add(&mut c.overflow, [o.as_str()], 0);
        }
        for o in hop::Outcome::ALL {
            add(&mut c.hop, [o.as_str()], 0);
        }
        add(&mut c.restricted_offbox, [], 0);
        c
    }

    fn time(&mut self, stage: Stage, d: Duration) {
        self.duration
            .entry(stage)
            .or_default()
            .observe(d.as_secs_f64());
    }

    fn render(&self, out: &mut Exposition) {
        render_counters(out, &REQUESTS, &self.requests);
        render_counters(out, &SHED, &self.shed);
        render_counters(out, &PICK, &self.pick);
        render_counters(out, &PICK_UNKNOWN, &self.pick_unknown);
        render_counters(out, &STICKY, &self.sticky);
        render_counters(out, &COMPLETED, &self.completed);
        render_counters(out, &UPSTREAM_ERRORS, &self.upstream_errors);
        render_counters(out, &PROMPT_TOKENS, &self.prompt_tokens);
        render_counters(out, &COMPLETION_TOKENS, &self.completion_tokens);
        render_counters(out, &OVERFLOW, &self.overflow);
        render_counters(out, &HOP, &self.hop);
        render_counters(out, &RESTRICTED_OFFBOX, &self.restricted_offbox);
        for (stage, h) in &self.duration {
            out.histogram(&DURATION, [stage.as_str()], h);
        }
    }
}

struct Inner {
    gate: Gate,
    fleet: Fleet,
    pool: String,
    shared_prefix_tokens: u64,
    counters: Mutex<Counters>,
    /// Read on every scrape; `None` reads as unavailable.
    overflow: Mutex<Option<Arc<Overflow>>>,
}

/// One gateway's metrics. Cloning is cheap and every clone counts into the same series.
#[derive(Clone)]
pub struct Metrics(Arc<Inner>);

impl std::fmt::Debug for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metrics")
            .field("pool", &self.0.pool)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Metrics {
    /// `pool` is the pool label on every replica gauge. `shared_prefix_tokens` is the token
    /// length of the prefix every prompt shares, which splits each response's cached tokens into
    /// the shared prefix and the run's own history.
    pub fn new(gate: Gate, fleet: Fleet, pool: &str, shared_prefix_tokens: u64) -> Self {
        Self(Arc::new(Inner {
            gate,
            fleet,
            pool: pool.to_owned(),
            shared_prefix_tokens,
            counters: Mutex::new(Counters::preregistered()),
            overflow: Mutex::new(None),
        }))
    }

    /// Makes `orch_overflow_backend_available` read the circuit.
    pub fn watch_overflow(&self, ov: Arc<Overflow>) {
        *lock(&self.0.overflow) = Some(ov);
    }

    /// Folds one finished request into the counters. Queue, upstream, hop and overflow time are
    /// observed only when the request spent any, so a refusal does not pull their quantiles
    /// toward 0.
    pub fn observe(&self, ev: &Event) {
        let mut c = lock(&self.0.counters);
        add(
            &mut c.requests,
            [ev.priority.as_str(), ev.class.as_str()],
            1,
        );
        c.time(Stage::Gateway, ev.gateway);
        c.time(Stage::E2e, ev.total);
        for (stage, d) in [
            (Stage::Queue, ev.queue),
            (Stage::Local, ev.upstream),
            (Stage::Hop, ev.hop_time),
            (Stage::Overflow, ev.overflow_time),
        ] {
            if !d.is_zero() {
                c.time(stage, d);
            }
        }
        if let Some(h) = ev.hop {
            add(&mut c.hop, [h.as_str()], 1);
        }
        let pod = ev.placement.as_ref().map_or("", |p| p.pod.as_str());
        if let Some(p) = &ev.placement {
            add(&mut c.sticky, [p.sticky.as_str()], 1);
            if !pod.is_empty() {
                add(&mut c.pick, [pod, p.policy.as_str()], 1);
                if p.unknown {
                    add(&mut c.pick_unknown, [pod], 1);
                }
            }
        }
        if let Some(o) = ev.overflow {
            add(&mut c.overflow, [o.as_str()], 1);
        }
        let offbox = ev.overflow == Some(overflow::Outcome::Sent);
        if offbox && ev.class == DataClass::Restricted {
            // Unreachable by construction (may_leave, the Offboxable permit, the backend's own
            // check); counted so the alert fires if it ever is.
            add(&mut c.restricted_offbox, [], 1);
        }

        // No status means the client went away before one was written.
        let Some(status) = ev.status else {
            return;
        };
        let code = status.to_string();
        if let Some(reason) = ev.reason {
            add(&mut c.shed, [reason.as_str(), &code], 1);
            if reason == Reason::UpstreamError {
                add(&mut c.upstream_errors, [pod, "transport"], 1);
            }
            return;
        }
        let finish = ev.usage.as_ref().map_or("", |u| u.finish_reason.as_str());
        if offbox {
            // Off the box there is no prefix cache to split tokens by, and no worker to charge
            // them to.
            add(&mut c.completed, [OVERFLOW_POD, &code, finish], 1);
            return;
        }
        add(&mut c.completed, [pod, &code, finish], 1);
        if status != 200 {
            add(&mut c.upstream_errors, [pod, "status"], 1);
            return;
        }
        let (prompt, cached, completion) = ev.usage.as_ref().map_or((0, 0, 0), |u| {
            (u.prompt_tokens, u.cached_tokens, u.completion_tokens)
        });
        let shared = self.0.shared_prefix_tokens;
        add(&mut c.prompt_tokens, [pod, "shared_hit"], cached.min(shared));
        add(&mut c.prompt_tokens, [pod, "run_hit"], cached.saturating_sub(shared));
        add(&mut c.prompt_tokens, [pod, "miss"], prompt.saturating_sub(cached));
        add(&mut c.completion_tokens, [pod], completion);
    }

    /// The Prometheus text exposition the gateway serves on `/metrics`.
    pub fn render(&self) -> String {
        let mut out = Exposition::default();
        lock(&self.0.counters).render(&mut out);
        let available = lock(&self.0.overflow)
            .as_ref()
            .is_some_and(|ov| ov.available());
        out.sample(&OVERFLOW_AVAILABLE, [], b2f(available));
        self.collect_replicas(&mut out);
        out.finish()
    }

    /// Reads the Gate and the Fleet once and emits every replica gauge. The two are joined by
    /// pod, so a pod the Fleet has not reported reads as Down.
    fn collect_replicas(&self, out: &mut Exposition) {
        let mut status: HashMap<String, WorkerStatus> = self
            .0
            .fleet
            .status()
            .into_iter()
            .map(|st| (st.pod.clone(), st))
            .collect();
        let pool = self.0.pool.as_str();
        let (mut free, mut capacity) = (0_u64, 0_u64);
        for ws in self.0.gate.view() {
            let pod = ws.view.pod.as_str();
            let st = status.remove(pod).unwrap_or_else(|| WorkerStatus {
                pod: pod.to_owned(),
                phase: Phase::Down,
                last_scrape: None,
                last_probe_s: 0.0,
                probes: 0,
                restarts: 0,
            });
            for (spec, read) in &REPLICA_GAUGES {
                out.sample(spec, [pool, pod], read(&ws, &st));
            }
            for phase in PHASES {
                out.sample(
                    &REPLICA_PHASE,
                    [pool, pod, phase.as_str()],
                    b2f(st.phase == phase),
                );
            }
            out.sample(&REPLICA_RESTARTS, [pool, pod], st.restarts as f64);
            if ws.ready {
                free = free.saturating_add(ws.view.free_tokens());
                capacity = capacity.saturating_add(ws.view.kv_pool_tokens);
            }
        }
        let ratio = if capacity > 0 {
            free as f64 / capacity as f64
        } else {
            0.0
        };
        out.sample(&KV_FREE, [pool], ratio);
    }
}

fn b2f(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests;
