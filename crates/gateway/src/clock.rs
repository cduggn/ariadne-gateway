//! The gateway's one clock. Decisions read it as an `ariadne_decide::Timestamp`, an offset from
//! the epoch the clock fixed when it was built, and anything printed for a human converts that
//! offset back to wall time with `wall`.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use ariadne_decide::Timestamp;

/// A cheap, cloneable clock. Every clone reads the same epoch.
#[derive(Clone, Debug)]
pub struct Clock {
    source: Source,
    wall_epoch: SystemTime,
}

#[derive(Clone, Debug)]
enum Source {
    /// tokio's monotonic clock, so `tokio::time::pause` and `advance` move it in tests.
    Tokio(tokio::time::Instant),
    Manual(Arc<Mutex<Duration>>),
}

impl Clock {
    /// The real clock, with its epoch at this call.
    pub fn system() -> Self {
        Self {
            source: Source::Tokio(tokio::time::Instant::now()),
            wall_epoch: SystemTime::now(),
        }
    }

    /// A clock that moves only when its `ManualClock` handle advances it. Its epoch is the wall
    /// time of this call.
    pub fn manual() -> (Self, ManualClock) {
        let offset = Arc::new(Mutex::new(Duration::ZERO));
        let clock = Self {
            source: Source::Manual(Arc::clone(&offset)),
            wall_epoch: SystemTime::now(),
        };
        (clock, ManualClock(offset))
    }

    pub fn now(&self) -> Timestamp {
        Timestamp::from_epoch_offset(match &self.source {
            Source::Tokio(epoch) => epoch.elapsed(),
            Source::Manual(offset) => *offset.lock().unwrap_or_else(PoisonError::into_inner),
        })
    }

    /// The wall time `t` corresponds to.
    pub fn wall(&self, t: Timestamp) -> SystemTime {
        self.wall_epoch
            .checked_add(t.epoch_offset())
            .unwrap_or(self.wall_epoch)
    }

    /// The time from now until `t`, zero when `t` is not in the future. tokio timers take this.
    pub fn until(&self, t: Timestamp) -> Duration {
        t.saturating_duration_since(self.now())
    }
}

/// Moves a `Clock::manual` clock forward.
#[derive(Clone, Debug)]
pub struct ManualClock(Arc<Mutex<Duration>>);

impl ManualClock {
    pub fn advance(&self, d: Duration) {
        let mut offset = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        *offset = offset.saturating_add(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_moves_only_when_advanced() {
        let (clock, hand) = Clock::manual();
        let start = clock.now();
        assert_eq!(start, Timestamp::from_epoch_offset(Duration::ZERO));
        hand.advance(Duration::from_millis(1500));
        let later = clock.clone().now();
        assert_eq!(
            later.saturating_duration_since(start),
            Duration::from_millis(1500)
        );
        assert_eq!(
            clock.wall(later).duration_since(clock.wall(start)).ok(),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(clock.until(start), Duration::ZERO);
        assert_eq!(
            clock.until(later.saturating_add(Duration::from_secs(2))),
            Duration::from_secs(2)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn system_clock_follows_paused_tokio_time() {
        let clock = Clock::system();
        let start = clock.now();
        tokio::time::advance(Duration::from_secs(3)).await;
        assert_eq!(
            clock.now().saturating_duration_since(start),
            Duration::from_secs(3)
        );
    }
}
