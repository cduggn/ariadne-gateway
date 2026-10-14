//! The gateway's stateful side. `Queue` is the per-worker in-flight cap and the two-lane wait list
//! behind it, the only step of the pipeline that waits.

mod gate;
mod queue;
pub mod scrape;

pub use gate::{
    Admission, Config, Decision, Gate, OffboxTicket, Outcome, SHARED_TENANT, Scrape, TenantLimit,
    Ticket,
};
pub use queue::{Depth, Queue, QueueRefusal, Slot};

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Locks `m` even when a panic poisoned it. Nothing under these locks panics by construction, and
/// a poisoned lock must not turn every later request, or a `Drop` running during unwinding, into
/// a second panic.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}
