use std::time::Duration;

use crate::request::{Request, RunId};
use crate::shed::{Policy, Reason, Refusal, Snap, TenantState, Verdict, WorkerView, should_shed};
use crate::time::Timestamp;

/// Which admitted worker `pick` chooses when more than one passes `should_shed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PickPolicy {
    /// Keeps a request on its bound worker while that worker is admitted and not much busier
    /// than the others, and otherwise takes the least loaded admitted worker. It is the default,
    /// because it keeps the run's cached history without pinning the run to a worker that has
    /// fallen far behind.
    PrefixThenLoad,
    /// Takes the least loaded admitted worker and ignores stickiness. It is the control arm of
    /// the stickiness A/B.
    LeastLoaded,
    /// Samples two admitted workers at random and takes the less loaded one.
    P2c,
}

impl PickPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PrefixThenLoad => "prefix_then_load",
            Self::LeastLoaded => "least_loaded",
            Self::P2c => "p2c",
        }
    }
}

/// Parses a policy flag value, trimming spaces and ignoring case. An unknown value is `None`, so
/// a typo fails startup instead of changing routing.
pub fn parse_pick_policy(s: &str) -> Option<PickPolicy> {
    [
        PickPolicy::PrefixThenLoad,
        PickPolicy::LeastLoaded,
        PickPolicy::P2c,
    ]
    .into_iter()
    .find(|p| s.trim().eq_ignore_ascii_case(p.as_str()))
}

/// The outcome label on `orch_sticky_total{outcome}`: what happened relative to the request's
/// bound worker. `pick` sets it even when it sheds, so the metric alone can score the
/// stickiness A/B.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sticky {
    /// The request has no run, so stickiness did not apply.
    None,
    /// The run has no bound worker yet. A successful placement creates the binding.
    New,
    /// `pick` chose the bound worker.
    Hit,
    /// The bound worker passed `should_shed` but was not chosen: it was more than
    /// `Fleet::stick_slack` busier than the best other worker, or the policy ignores stickiness.
    BrokenLoad,
    /// The bound worker was a candidate and `should_shed` refused it.
    BrokenShed,
    /// The bound worker was never a candidate: it was not ready, it was down, or the staleness
    /// floor dropped it.
    BrokenGone,
}

impl Sticky {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::New => "new",
            Self::Hit => "hit",
            Self::BrokenLoad => "broken_load",
            Self::BrokenShed => "broken_shed",
            Self::BrokenGone => "broken_gone",
        }
    }
}

/// One worker as `pick` sees it. Each worker has its own `age`, the time since its last
/// successful scrape, because workers go stale independently.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerState {
    pub view: WorkerView,
    /// The warm-up gate has passed.
    pub ready: bool,
    pub age: Duration,
}

/// `stale_after` and `down_after` separate fresh, stale and down workers. `capacity_floor` is
/// the share of the eligible fleet's `max_inflight` that must stay fresh before `pick` drops
/// stale workers. Below it, stale workers stay and `pick` uses their last-known telemetry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Staleness {
    pub stale_after: Duration,
    pub down_after: Duration,
    pub capacity_floor: f64,
}

pub const DEFAULT_STALENESS: Staleness = Staleness {
    stale_after: Duration::from_secs(2),
    down_after: Duration::from_secs(10),
    capacity_floor: 0.75,
};

impl WorkerView {
    /// The course's `score_load`: in-flight plus queued requests as a fraction of the worker's
    /// cap, plus the fraction of KV in use. The gateway's own counts replace vLLM's running and
    /// waiting, and KV in use includes the gateway's reservations.
    pub fn load(&self) -> f64 {
        self.occupancy() + self.kv_in_use()
    }

    /// `load` with `tokens` of the worker's KV left out of the KV term. `pick` uses it to score
    /// a run's bound worker without the run's own history, which is load that moves with the
    /// run and so should not count against keeping it. A pool of 0 discounts nothing.
    pub fn load_without(&self, tokens: u64) -> f64 {
        let mut kv = self.kv_in_use();
        if self.kv_pool_tokens > 0 {
            kv -= tokens as f64 / self.kv_pool_tokens as f64;
        }
        self.occupancy() + if kv < 0.0 { 0.0 } else { kv }
    }

    /// An unconfigured worker with a cap of 0 divides by 1.
    fn occupancy(&self) -> f64 {
        (self.in_flight as f64 + self.queued as f64) / self.max_inflight.max(1) as f64
    }

    fn kv_in_use(&self) -> f64 {
        1.0 - self.free_ratio()
    }
}

/// 0.25 load units is four of the default sixteen in-flight slots.
pub const DEFAULT_STICK_SLACK: f64 = 0.25;

/// Everything `pick` reads about a request's candidate workers.
#[derive(Debug, Clone, PartialEq)]
pub struct Fleet {
    pub now: Timestamp,
    pub workers: Vec<WorkerState>,
    pub tenant: TenantState,
    pub est_tokens: u64,
    /// The pod this request's run is bound to.
    pub bound: Option<String>,
    /// Prompt tokens of the run's last settled step, held in `bound`'s KV; 0 if unknown.
    pub history: u64,
    /// Admission thresholds passed through to `should_shed`.
    pub shed: Policy,
    pub staleness: Staleness,
    /// Load units the bound worker may exceed the best other worker by.
    pub stick_slack: f64,
}

/// The worker `pick` chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placed {
    pub pod: String,
    /// The chosen worker's telemetry was stale.
    pub unknown: bool,
}

/// `pick`'s result. `policy` and `sticky` are metric labels and are set whether or not a worker
/// was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub result: Result<Placed, Refusal>,
    pub policy: PickPolicy,
    pub sticky: Sticky,
}

impl Placement {
    pub fn placed(&self) -> bool {
        self.result.is_ok()
    }

    pub fn pod(&self) -> Option<&str> {
        self.result.as_ref().ok().map(|p| p.pod.as_str())
    }

    pub fn unknown(&self) -> bool {
        self.result.as_ref().is_ok_and(|p| p.unknown)
    }
}

/// The course's `pick(req, workers, *, policy)`. It keeps ready workers that are not down and
/// applies the staleness floor. It then runs `should_shed` on each candidate, so routing and
/// admission cannot disagree. Last, it chooses among the admitted workers by policy.
///
/// `rnd(n)` returns a value in `[0, n)`. Only `PickPolicy::P2c` calls it, and only when more
/// than one worker is admitted.
pub fn pick(
    r: &Request,
    f: &Fleet,
    policy: PickPolicy,
    rnd: impl FnMut(usize) -> usize,
) -> Placement {
    let eligible: Vec<&WorkerState> = f
        .workers
        .iter()
        .filter(|w| w.ready && w.age < f.staleness.down_after)
        .collect();
    let ready = eligible.len();
    let remaining = drop_stale_below_floor(eligible, &f.staleness);
    let (admitted, refusals) = admit_candidates(r, f, &remaining);

    let bound = f.bound.as_deref();
    let sticky = |chosen| classify_sticky(bound, chosen, r.run.as_ref(), &remaining, &admitted);

    let Some(&first) = admitted.first() else {
        // Every candidate refused, or there were none to ask.
        let refusal = earliest_gate(refusals).unwrap_or_else(|| Refusal {
            reason: Reason::NoEligiblePod,
            retry_after: Duration::from_secs(2),
            detail: format!(
                "{} workers, {ready} ready, none survive staleness filtering",
                f.workers.len()
            ),
        });
        return Placement {
            result: Err(refusal),
            policy,
            sticky: sticky(None),
        };
    };

    let chosen = match policy {
        PickPolicy::PrefixThenLoad => {
            bound_within_slack(f, &admitted).unwrap_or_else(|| least_loaded(first, &admitted))
        }
        PickPolicy::LeastLoaded => least_loaded(first, &admitted),
        PickPolicy::P2c => sample_two(first, &admitted, rnd),
    };
    Placement {
        sticky: sticky(Some(&chosen.view.pod)),
        result: Ok(Placed {
            pod: chosen.view.pod.clone(),
            unknown: chosen.age >= f.staleness.stale_after,
        }),
        policy,
    }
}

/// Drops stale workers only if the fresh ones hold at least `capacity_floor` of the eligible
/// fleet's `max_inflight`.
fn drop_stale_below_floor<'a>(
    mut eligible: Vec<&'a WorkerState>,
    st: &Staleness,
) -> Vec<&'a WorkerState> {
    let is_stale = |w: &WorkerState| w.age >= st.stale_after;
    let capacity = |w: &&WorkerState| w.view.max_inflight as f64;

    let total: f64 = eligible.iter().map(capacity).sum();
    let fresh: f64 = eligible.iter().filter(|w| !is_stale(w)).map(capacity).sum();
    let has_stale = eligible.iter().any(|w| is_stale(w));

    if has_stale && total > 0.0 && fresh >= st.capacity_floor * total {
        eligible.retain(|w| !is_stale(w));
    }
    eligible
}

/// Runs `should_shed` on every remaining worker. Only the run's bound worker gets the resident
/// exemption.
fn admit_candidates<'a>(
    r: &Request,
    f: &Fleet,
    remaining: &[&'a WorkerState],
) -> (Vec<&'a WorkerState>, Vec<Refusal>) {
    let mut admitted = Vec::new();
    let mut refusals = Vec::new();
    for &w in remaining {
        let snap = Snap {
            now: f.now,
            tenant: f.tenant,
            worker: &w.view,
            resident: f.bound.as_deref() == Some(w.view.pod.as_str()),
            est_tokens: f.est_tokens,
            policy: f.shed,
        };
        match should_shed(r, &snap) {
            Verdict::Admit => admitted.push(w),
            Verdict::Shed(refusal) => refusals.push(refusal),
        }
    }
    (admitted, refusals)
}

/// When candidates refuse for different reasons, the earliest gate in `should_shed`'s order
/// wins, so a 429 outranks a 503 from another worker.
fn earliest_gate(refusals: Vec<Refusal>) -> Option<Refusal> {
    refusals.into_iter().min_by_key(|v| match v.reason {
        Reason::TenantTokens => 0,
        Reason::KvFree => 1,
        Reason::TimeoutQueue => 2,
        Reason::P99Spread => 3,
        // Not `should_shed` gates, so they never reach here.
        Reason::NoEligiblePod | Reason::QueueFull => 4,
    })
}

/// The `Sticky` outcome of `chosen` relative to `bound`. With nothing chosen, `admitted` is
/// empty and the outcome says whether the bound worker was a candidate at all.
fn classify_sticky(
    bound: Option<&str>,
    chosen: Option<&str>,
    run: Option<&RunId>,
    remaining: &[&WorkerState],
    admitted: &[&WorkerState],
) -> Sticky {
    let holds = |ws: &[&WorkerState], pod: &str| ws.iter().any(|w| w.view.pod == pod);
    match (bound, run) {
        (None, None) => Sticky::None,
        (None, Some(_)) => Sticky::New,
        (Some(bound), _) if chosen == Some(bound) => Sticky::Hit,
        (Some(bound), _) if holds(admitted, bound) => Sticky::BrokenLoad,
        (Some(bound), _) if holds(remaining, bound) => Sticky::BrokenShed,
        (Some(_), _) => Sticky::BrokenGone,
    }
}

/// The bound worker, if it is admitted and within `stick_slack` load units of the best other
/// admitted worker.
///
/// It is scored without the run's own history, because that KV moves with the run: it is on the
/// bound worker only because the run is there, and keeping the run costs nothing extra.
/// Counting it made one long run on an idle fleet break its own stickiness and bounce between
/// workers.
fn bound_within_slack<'a>(f: &Fleet, admitted: &[&'a WorkerState]) -> Option<&'a WorkerState> {
    let pod = f.bound.as_deref()?;
    let bound = admitted.iter().copied().find(|w| w.view.pod == pod)?;
    let best_other = admitted
        .iter()
        .filter(|w| w.view.pod != pod)
        .map(|w| w.view.load())
        .reduce(|min, load| if load < min { load } else { min });
    best_other
        .is_none_or(|min| bound.view.load_without(f.history) <= min + f.stick_slack)
        .then_some(bound)
}

fn least_loaded<'a>(first: &'a WorkerState, admitted: &[&'a WorkerState]) -> &'a WorkerState {
    admitted.iter().copied().fold(first, less_loaded)
}

/// Samples two distinct admitted workers and takes the less loaded one. With one admitted
/// worker it takes that worker and never calls `rnd`.
fn sample_two<'a>(
    first: &'a WorkerState,
    admitted: &[&'a WorkerState],
    mut rnd: impl FnMut(usize) -> usize,
) -> &'a WorkerState {
    let n = admitted.len();
    if n < 2 {
        return first;
    }
    // Clamped so a misbehaving `rnd` cannot index out of range.
    let i = rnd(n).min(n - 1);
    let mut j = rnd(n - 1).min(n - 2);
    if j >= i {
        j += 1;
    }
    less_loaded(admitted[i], admitted[j])
}

/// Ties go to the lower pod name, so the choice is deterministic.
fn less_loaded<'a>(best: &'a WorkerState, w: &'a WorkerState) -> &'a WorkerState {
    let (load, best_load) = (w.view.load(), best.view.load());
    if load < best_load || (load == best_load && w.view.pod < best.view.pod) {
        w
    } else {
        best
    }
}
