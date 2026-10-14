use std::time::Duration;

use ariadne_decide::{DataClass, PickPolicy, Priority, RejectReason, RunId, Sticky};

use crate::hop;
use crate::overflow;

/// One request as the gateway saw it, from arrival to the last byte written. The serve module
/// fills one per request, including refusals and 400s, logs it and hands it to the metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// `X-Request-Id` verbatim.
    pub request_id: String,
    pub run: Option<RunId>,
    pub step: u64,
    pub tenant: String,
    pub priority: Priority,
    pub class: DataClass,
    /// What placement decided. `None` when the guard refused the request.
    pub placement: Option<Placed>,
    /// `None` when the client went away before anything was written.
    pub status: Option<u16>,
    pub reason: Option<Reason>,
    /// Time in this process outside the queue, the hop and the upstream call.
    pub gateway: Duration,
    pub queue: Duration,
    /// Upstream time; covers both attempts after a hop fallback.
    pub upstream: Duration,
    pub total: Duration,
    /// Set only for a 200 that carried usage.
    pub usage: Option<Usage>,
    /// Set only for a 503 the gateway could not serve. `Sent` means the backend's 200 was
    /// relayed and `placement.pod` is empty.
    pub overflow: Option<overflow::Outcome>,
    pub overflow_time: Duration,
    /// Set only when the hop is configured and the request moved a run off a worker that still
    /// held its history.
    pub hop: Option<hop::Outcome>,
    pub hop_time: Duration,
    /// The new worker did not answer the hopped request and the gateway sent it again without
    /// the hop.
    pub hop_fallback: bool,
}

/// Placement's labels for one request. `pod` is empty when no worker was chosen.
#[derive(Debug, Clone, PartialEq)]
pub struct Placed {
    pub pod: String,
    pub policy: PickPolicy,
    pub sticky: Sticky,
    /// The chosen worker's telemetry was stale.
    pub unknown: bool,
    pub est: u64,
}

/// The token usage a worker or the overflow backend reported in a 200.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    pub finish_reason: String,
}

/// Why a request did not get a worker's answer: the `reason` field of the request log line and
/// the `reason` label of `orch_shed_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    Guard(RejectReason),
    Shed(ariadne_decide::Reason),
    ClientGone,
    UpstreamError,
}

impl Reason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Guard(r) => r.as_str(),
            Self::Shed(r) => r.as_str(),
            Self::ClientGone => "client_gone",
            Self::UpstreamError => "upstream_error",
        }
    }
}
