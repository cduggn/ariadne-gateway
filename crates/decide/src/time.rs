use std::time::Duration;

/// A point in time, held as the offset from an epoch the caller fixes once (process start, say).
///
/// The core never reads a clock, and `std::time::Instant` cannot be built in a test, so the shell
/// converts its monotonic readings to this before calling in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(Duration);

impl Timestamp {
    pub const fn from_epoch_offset(offset: Duration) -> Self {
        Self(offset)
    }

    pub const fn epoch_offset(self) -> Duration {
        self.0
    }

    pub const fn saturating_add(self, d: Duration) -> Self {
        Self(self.0.saturating_add(d))
    }

    /// Time elapsed since `earlier`, or zero when `earlier` is not in the past.
    pub const fn saturating_duration_since(self, earlier: Self) -> Duration {
        self.0.saturating_sub(earlier.0)
    }
}
