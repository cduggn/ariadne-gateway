use std::fmt;
use std::time::Duration;

use crate::request::{Priority, Request};
use crate::time::Timestamp;

/// Why a request was refused: the `reason` label on `orch_shed_total{reason,code}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    /// The request would overdraw its tenant's token bucket.
    TenantTokens,
    /// The worker's effective free KV would fall below the admission line, or below the
    /// resident floor for a run already there.
    KvFree,
    /// The projected wait in the worker's queue would take the request past its deadline.
    TimeoutQueue,
    /// The worker's tail latency has moved far from its median while its queue keeps growing.
    /// Batch requests only.
    P99Spread,
    /// No worker survived `pick`'s ready and staleness filters.
    NoEligiblePod,
    /// The worker's queue was full, or an interactive arrival displaced this batch request.
    QueueFull,
}

impl Reason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TenantTokens => "tenant_tokens",
            Self::KvFree => "kv_free",
            Self::TimeoutQueue => "timeout_queue",
            Self::P99Spread => "p99_spread",
            Self::NoEligiblePod => "no_eligible_pod",
            Self::QueueFull => "queue_full",
        }
    }

    /// The status the client sees. It understands only these two: 429 for its own budget, 503
    /// for the fleet's capacity.
    pub const fn code(self) -> u16 {
        match self {
            Self::TenantTokens => 429,
            Self::KvFree
            | Self::TimeoutQueue
            | Self::P99Spread
            | Self::NoEligiblePod
            | Self::QueueFull => 503,
        }
    }
}

/// One refusal: the reason, the retry hint and a human-readable explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub reason: Reason,
    pub retry_after: Duration,
    pub detail: String,
}

impl Refusal {
    pub const fn code(&self) -> u16 {
        self.reason.code()
    }

    /// Whether the refusal must be answered locally and kept away from the overflow decision.
    /// Only a 503 may leave the box.
    pub const fn stays(&self) -> bool {
        self.code() != 503
    }
}

/// `should_shed`'s result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Admit,
    Shed(Refusal),
}

/// One worker's state as the gate sees it: the last scrape combined with the gateway's own
/// reservations since that scrape. It has no clock, because `pick` judges staleness and
/// `should_shed` does not.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WorkerView {
    pub pod: String,
    pub kv_pool_tokens: u64,
    /// Scraped fraction of the KV pool in use, 0..1.
    pub kv_usage: f64,
    /// The gateway's estimate for requests dispatched here since the last scrape.
    pub reserved_tokens: u64,
    pub in_flight: usize,
    pub queued: usize,
    pub max_inflight: usize,
    /// Recent mean seconds per request on this pod; 0 if unknown.
    pub mean_service_s: f64,
    pub ttft_p50_s: f64,
    pub ttft_p99_s: f64,
    pub waiting_rising: bool,
}

impl WorkerView {
    /// Effective free KV: the scraped free pool minus the tokens the gateway reserved on this
    /// worker since the last scrape.
    pub fn free_tokens(&self) -> u64 {
        let scraped_free = (self.kv_pool_tokens as f64 * (1.0 - self.kv_usage)).round() as u64;
        scraped_free.saturating_sub(self.reserved_tokens)
    }

    /// `free_tokens` as a fraction of the pool. An unknown or misconfigured worker with a pool
    /// of 0 reports 0.
    pub fn free_ratio(&self) -> f64 {
        if self.kv_pool_tokens == 0 {
            return 0.0;
        }
        self.free_tokens() as f64 / self.kv_pool_tokens as f64
    }
}

/// One tenant's token bucket at decision time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TenantState {
    pub available_tokens: f64,
    pub rate_per_s: f64,
}

/// The admission thresholds. There is no `Default`: start from `DEFAULT_POLICY`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    /// Fraction of the KV pool in use past which a new run is refused, e.g. 0.80.
    pub kv_line: f64,
    /// Lower free-ratio floor granted to a resident run, e.g. 0.05.
    pub resident_floor: f64,
    /// p99/p50 multiple past which a batch worker looks unstable, e.g. 4.
    pub spread_ratio: f64,
}

pub const DEFAULT_POLICY: Policy = Policy {
    kv_line: 0.80,
    resident_floor: 0.05,
    spread_ratio: 4.0,
};

/// Everything `should_shed` reads about one candidate placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap<'a> {
    pub now: Timestamp,
    pub tenant: TenantState,
    pub worker: &'a WorkerView,
    /// This request's run is bound to `worker`.
    pub resident: bool,
    /// Prompt estimate plus `Request::max_out`.
    pub est_tokens: u64,
    pub policy: Policy,
}

/// The course's `should_shed(req, snap)`. It checks one candidate worker without side effects.
/// The gates run in this order and the first refusal wins, so a request is never shed for two
/// reasons at once.
///
///  1. `tenant_tokens` (429): the tenant's bucket cannot cover the estimate.
///  2. `kv_free` (503): the worker's free KV is below the admission line, or below the resident
///     floor for a run continuing on this worker.
///  3. `timeout_queue` (503): the projected wait would pass the deadline.
///  4. `p99_spread` (503), batch only: the worker's tail latency has moved far from its median
///     while its queue keeps growing.
pub fn should_shed(r: &Request, s: &Snap<'_>) -> Verdict {
    tenant_tokens(s)
        .or_else(|| kv_free(s))
        .or_else(|| timeout_queue(r, s))
        .or_else(|| match r.priority {
            Priority::Batch => p99_spread(s.worker, &s.policy),
            Priority::Interactive => None,
        })
        .map_or(Verdict::Admit, Verdict::Shed)
}

fn tenant_tokens(s: &Snap<'_>) -> Option<Refusal> {
    let TenantState {
        available_tokens,
        rate_per_s,
    } = s.tenant;
    (s.est_tokens as f64 > available_tokens).then(|| Refusal {
        reason: Reason::TenantTokens,
        retry_after: tenant_retry_after(s.est_tokens, s.tenant),
        detail: format!(
            "need {} tokens, tenant has {} available at {}/s",
            s.est_tokens,
            GoFloat(available_tokens),
            GoFloat(rate_per_s),
        ),
    })
}

/// Time until the bucket refills enough to cover the shortfall, kept between 1s and 60s. A
/// tenant with no refill rate gets 60s, because there is nothing to estimate from.
fn tenant_retry_after(est_tokens: u64, tenant: TenantState) -> Duration {
    if tenant.rate_per_s <= 0.0 {
        return Duration::from_secs(60);
    }
    let deficit = est_tokens as f64 - tenant.available_tokens;
    let seconds = (deficit / tenant.rate_per_s).ceil().clamp(1.0, 60.0);
    Duration::from_secs(seconds as u64)
}

/// A run whose history is on this worker may continue down to the resident floor. Any other
/// request sheds at the admission line, or when its estimate does not fit in the free KV.
fn kv_free(s: &Snap<'_>) -> Option<Refusal> {
    let refuse = |detail: String| Refusal {
        reason: Reason::KvFree,
        retry_after: Duration::from_secs(2),
        detail,
    };
    let free = s.worker.free_ratio();

    if s.resident {
        let floor = s.policy.resident_floor;
        return (free < floor).then(|| {
            refuse(format!(
                "resident free ratio {} below floor {}",
                GoFloat(free),
                GoFloat(floor)
            ))
        });
    }

    let line = 1.0 - s.policy.kv_line;
    if free < line {
        return Some(refuse(format!(
            "free ratio {} below line {}",
            GoFloat(free),
            GoFloat(line)
        )));
    }
    let free_tokens = s.worker.free_tokens();
    (s.est_tokens > free_tokens).then(|| {
        refuse(format!(
            "estimate {} exceeds free tokens {free_tokens}",
            s.est_tokens
        ))
    })
}

/// The projected wait is 0 while the worker has a free slot. When it is full, the wait is the
/// queue ahead times the worker's mean service time. A worker with no service history counts as
/// no wait, because a guess could shed a request that would have finished in time.
fn timeout_queue(r: &Request, s: &Snap<'_>) -> Option<Refusal> {
    let w = s.worker;
    let wait = if w.in_flight >= w.max_inflight && w.mean_service_s > 0.0 {
        let ahead_of = w.queued as f64 + 1.0;
        Duration::from_nanos((ahead_of * w.mean_service_s * 1e9) as u64)
    } else {
        Duration::ZERO
    };

    (s.now.saturating_add(wait) > r.deadline).then(|| Refusal {
        reason: Reason::TimeoutQueue,
        retry_after: Duration::from_secs(1),
        detail: format!(
            "projected wait {}s past deadline, {}/{} in flight",
            GoFloat(wait.as_secs_f64()),
            w.in_flight,
            w.max_inflight,
        ),
    })
}

/// A worker with no p50 history or a shrinking queue never sheds here.
fn p99_spread(w: &WorkerView, p: &Policy) -> Option<Refusal> {
    let unstable =
        w.ttft_p50_s > 0.0 && w.ttft_p99_s > p.spread_ratio * w.ttft_p50_s && w.waiting_rising;
    unstable.then(|| Refusal {
        reason: Reason::P99Spread,
        retry_after: Duration::from_secs(5),
        detail: format!(
            "p99 {}s exceeds {}x p50 {}s with a rising queue",
            GoFloat(w.ttft_p99_s),
            GoFloat(p.spread_ratio),
            GoFloat(w.ttft_p50_s),
        ),
    })
}

/// Renders a float as Go's `strconv.FormatFloat(f, 'f', -1, 64)` did, because the detail strings
/// reach the client in the error body. Rust's `Display` already agrees on every finite value
/// and on NaN; only the infinities differ.
struct GoFloat(f64);

impl fmt::Display for GoFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == f64::INFINITY {
            f.write_str("+Inf")
        } else if self.0 == f64::NEG_INFINITY {
            f.write_str("-Inf")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

/// The token usage vLLM reported for a run's previous step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RunUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub body_bytes: usize,
}

/// Estimates a request's tokens without a tokenizer. A run's first step uses body bytes divided
/// by 3.5. Later steps start from vLLM's usage for the previous step and add the new bytes
/// divided by 3.5. The result includes `max_out`, because the tenant's budget and the worker's
/// KV must cover the output too.
pub fn estimate_tokens(r: &Request, prior: Option<&RunUsage>) -> u64 {
    let prompt = match prior {
        None => bytes_to_tokens(r.body_bytes),
        Some(prior) => prior
            .prompt_tokens
            .saturating_add(prior.completion_tokens)
            .saturating_add(bytes_to_tokens(
                r.body_bytes.saturating_sub(prior.body_bytes),
            )),
    };
    prompt.saturating_add(r.max_out)
}

fn bytes_to_tokens(bytes: usize) -> u64 {
    (bytes as f64 / 3.5).ceil() as u64
}
