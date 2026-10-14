//! The gateway's HTTP shell: guard, admit, forward, relay.

mod event;

pub use event::{Event, Placed, Reason, Usage};
