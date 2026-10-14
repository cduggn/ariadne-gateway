//! Sends a 503 the gateway cannot serve to a backend off the box, when the
//! stay-or-leave rule allows it. It is off unless configured, and every
//! failure falls back to the 503 the client would have had anyway, so
//! overflow can turn a refusal into an answer but never makes one worse.
//!
//! `Backend` is the wire client: `Superlinked` speaks the OpenAI-compatible
//! chat-completions shape over HTTPS with a bearer key, and `Null` is the
//! backend of an unconfigured gateway. `Overflow` wraps a backend with the
//! circuit serve calls: probes mark the backend available or not, a send
//! that fails on the backend's side opens the circuit until the next probe
//! passes, and an in-flight cap keeps a burst of 503s from becoming a burst
//! of paid calls. Whether a request may leave at all is
//! `ariadne_decide::may_leave`, and every send takes the `Offboxable` only it
//! can mint.

mod superlinked;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use ariadne_decide::{OVERFLOW_BLOCKED, OVERFLOW_NO_BACKEND, Offboxable};
use bytes::Bytes;
use http::{HeaderValue, StatusCode};
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use self::superlinked::Fault;
pub use self::superlinked::{ApiKey, Config, ConfigError, InvalidConfig, Superlinked};

/// The `result` label of `orch_overflow_total` and the `X-Overflow` header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// A 503 for a restricted request was kept on the box. Decide's label.
    Blocked,
    /// The request could leave but no backend is configured. Decide's label.
    NoBackend,
    /// The backend answered 200 and its answer was relayed.
    Sent,
    /// The backend was tried and gave an error, a non-200 or no answer in
    /// time, so the client got the original 503.
    Failed,
    /// The circuit was open, so the backend was not tried.
    Unavailable,
    /// `max_inflight` overflow calls were already running.
    Capped,
    /// The tenant's token bucket could not cover the estimate, so the
    /// request stayed on the box with its 503. Serve sets this one.
    Quota,
}

impl Outcome {
    /// Every outcome, for pre-registering the metric's labels, in Go's order.
    pub const ALL: [Self; 7] = [
        Self::Blocked,
        Self::NoBackend,
        Self::Sent,
        Self::Failed,
        Self::Unavailable,
        Self::Capped,
        Self::Quota,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Blocked => OVERFLOW_BLOCKED,
            Self::NoBackend => OVERFLOW_NO_BACKEND,
            Self::Sent => "sent",
            Self::Failed => "failed",
            Self::Unavailable => "unavailable",
            Self::Capped => "capped",
            Self::Quota => "quota",
        }
    }
}

/// A backend's reply, read up to its size cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    pub status: StatusCode,
    pub content_type: Option<HeaderValue>,
    pub body: Bytes,
}

/// Why a send did not produce a 200.
#[derive(Debug)]
pub enum SendError {
    /// The `Null` backend answers this.
    NoBackend,
    /// The body is not a JSON object, so the model could not be rewritten.
    Body(serde_json::Error),
    Transport(reqwest::Error),
    /// No answer within the configured timeout.
    Timeout {
        url: String,
    },
    /// The backend answered a status other than 200.
    Status(StatusCode),
}

impl fmt::Display for SendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackend => f.write_str("overflow: no backend configured"),
            Self::Body(err) => write!(f, "overflow: body is not a JSON object: {err}"),
            Self::Transport(err) => write!(f, "{err}"),
            Self::Timeout { url } => write!(f, "Post {url:?}: context deadline exceeded"),
            Self::Status(status) => write!(f, "overflow: backend answered {}", status.as_u16()),
        }
    }
}

impl std::error::Error for SendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Body(err) => Some(err),
            Self::Transport(err) => Some(err),
            Self::NoBackend | Self::Timeout { .. } | Self::Status(_) => None,
        }
    }
}

impl From<reqwest::Error> for SendError {
    fn from(err: reqwest::Error) -> Self {
        Self::Transport(err)
    }
}

/// Why a probe failed.
#[derive(Debug)]
pub enum ProbeError {
    NoBackend,
    Transport(reqwest::Error),
    /// No answer within `PROBE_TIMEOUT`.
    Timeout,
    /// The probe path answered outside 2xx.
    Status {
        path: String,
        status: StatusCode,
    },
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackend => f.write_str("overflow: no backend configured"),
            Self::Transport(err) => write!(f, "{err}"),
            Self::Timeout => f.write_str("context deadline exceeded"),
            Self::Status { path, status } => {
                write!(f, "probe {path} answered {}", status.as_u16())
            }
        }
    }
}

impl std::error::Error for ProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(err) => Some(err),
            Self::NoBackend | Self::Timeout | Self::Status { .. } => None,
        }
    }
}

impl From<reqwest::Error> for ProbeError {
    fn from(err: reqwest::Error) -> Self {
        Self::Transport(err)
    }
}

/// One place a 503 may be sent. The gateway picks one at startup.
#[derive(Debug)]
pub enum Backend {
    /// An unconfigured gateway's backend: every call fails, so the client
    /// always gets the 503.
    Null,
    Superlinked(Superlinked),
}

impl Backend {
    /// Labels the backend in logs and headers.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Superlinked(_) => "superlinked",
        }
    }

    /// Posts the request body and returns the backend's answer.
    pub async fn send(&self, permit: Offboxable, body: Bytes) -> Result<Answer, SendError> {
        match self {
            Self::Null => Err(SendError::NoBackend),
            Self::Superlinked(backend) => backend.send(permit, body).await,
        }
    }

    /// Checks that the backend is reachable and the credentials work.
    pub async fn probe(&self) -> Result<(), ProbeError> {
        match self {
            Self::Null => Err(ProbeError::NoBackend),
            Self::Superlinked(backend) => backend.probe().await,
        }
    }
}

/// What `Overflow::send` did. Only `Sent` carries an answer to relay, and
/// its status is always 200; every other case answers the original 503.
#[derive(Debug)]
pub enum Attempt {
    Sent {
        content_type: Option<HeaderValue>,
        body: Bytes,
        took: Duration,
    },
    Failed {
        took: Duration,
        error: SendError,
    },
    Unavailable,
    Capped,
}

impl Attempt {
    pub fn outcome(&self) -> Outcome {
        match self {
            Self::Sent { .. } => Outcome::Sent,
            Self::Failed { .. } => Outcome::Failed,
            Self::Unavailable => Outcome::Unavailable,
            Self::Capped => Outcome::Capped,
        }
    }

    /// The time spent with the backend, zero unless it was called.
    pub fn took(&self) -> Duration {
        match self {
            Self::Sent { took, .. } | Self::Failed { took, .. } => *took,
            Self::Unavailable | Self::Capped => Duration::ZERO,
        }
    }
}

/// A log line's level, as slog names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

/// Receives log events: level, message, then the attributes in order.
pub type Log = Arc<dyn Fn(Level, &str, &[(&str, Value)]) + Send + Sync>;

/// A monotonic reading, used only to measure `took`.
pub type Clock = Arc<dyn Fn() -> Duration + Send + Sync>;

/// Bounds one probe. It is short because a probe runs on the gateway's
/// clock, not a request's.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// The circuit over one backend. It starts open: unavailable until a probe
/// passes.
pub struct Overflow {
    backend: Backend,
    interval: Duration,
    now: Clock,
    log: Log,
    up: AtomicBool,
    probes: AtomicU64,
    slots: Semaphore,
}

impl fmt::Debug for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Overflow")
            .field("backend", &self.backend)
            .field("interval", &self.interval)
            .field("up", &self.up)
            .field("probes", &self.probes)
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}

impl Overflow {
    /// `interval` is how often `run` re-probes, zero for never;
    /// `max_inflight` caps concurrent sends, at least 1.
    pub fn new(
        backend: Backend,
        interval: Duration,
        max_inflight: usize,
        now: Clock,
        log: Log,
    ) -> Self {
        Self {
            backend,
            interval,
            now,
            log,
            up: AtomicBool::new(false),
            probes: AtomicU64::new(0),
            slots: Semaphore::new(max_inflight.max(1)),
        }
    }

    pub fn name(&self) -> &'static str {
        self.backend.name()
    }

    /// Whether the circuit is closed: the last probe passed and no send has
    /// failed on the backend's side since.
    pub fn available(&self) -> bool {
        self.up.load(Ordering::SeqCst)
    }

    /// How many probes have completed, passed or not.
    pub fn probes(&self) -> u64 {
        self.probes.load(Ordering::SeqCst)
    }

    /// Runs one probe, sets the availability from it and returns it. A probe
    /// never fails the gateway: a backend that is down, unreachable or
    /// refusing the key just stays unavailable.
    pub async fn probe(&self) -> bool {
        let result = tokio::time::timeout(PROBE_TIMEOUT, self.backend.probe())
            .await
            .unwrap_or(Err(ProbeError::Timeout));
        let was = self.up.swap(result.is_ok(), Ordering::SeqCst);
        self.probes.fetch_add(1, Ordering::SeqCst);
        let backend = ("backend", Value::from(self.name()));
        match (&result, was) {
            (Err(err), true) => (self.log)(
                Level::Warn,
                "overflow backend unavailable",
                &[backend, ("err", Value::from(err.to_string()))],
            ),
            (Err(err), false) => (self.log)(
                Level::Debug,
                "overflow backend still unavailable",
                &[backend, ("err", Value::from(err.to_string()))],
            ),
            (Ok(()), false) => (self.log)(Level::Info, "overflow backend available", &[backend]),
            (Ok(()), true) => {}
        }
        result.is_ok()
    }

    /// Probes once, then every interval until `shutdown` is cancelled. A
    /// probe in flight at shutdown is dropped and leaves the state as it was.
    pub async fn run(&self, shutdown: CancellationToken) {
        let schedule = async {
            self.probe().await;
            if self.interval.is_zero() {
                return std::future::pending().await;
            }
            let start = tokio::time::Instant::now() + self.interval;
            let mut ticks = tokio::time::interval_at(start, self.interval);
            ticks.set_missed_tick_behavior(MissedTickBehavior::Skip);
            loop {
                ticks.tick().await;
                self.probe().await;
            }
        };
        tokio::select! {
            () = shutdown.cancelled() => {}
            () = schedule => {}
        }
    }

    /// Forwards `body` when the circuit is closed and a slot is free. A send
    /// that fails on the backend's side opens the circuit until the next
    /// probe passes, so a backend that is down or out of credit costs at most
    /// one call per probe interval. Dropping the future (the client left)
    /// frees the slot and leaves the circuit alone.
    pub async fn send(&self, permit: Offboxable, body: Bytes) -> Attempt {
        if !self.available() {
            return Attempt::Unavailable;
        }
        let Ok(_slot) = self.slots.try_acquire() else {
            return Attempt::Capped;
        };
        let start = (self.now)();
        let result = self.backend.send(permit, body).await;
        let took = (self.now)().saturating_sub(start);
        let result = result.and_then(|answer| match answer.status {
            StatusCode::OK => Ok(answer),
            status => Err(SendError::Status(status)),
        });
        match result {
            Ok(answer) => Attempt::Sent {
                content_type: answer.content_type,
                body: answer.body,
                took,
            },
            Err(error) => {
                match error.fault() {
                    Fault::Backend => self.up.store(false, Ordering::SeqCst),
                    Fault::Request => {}
                }
                Attempt::Failed { took, error }
            }
        }
    }
}

#[cfg(test)]
mod tests;
