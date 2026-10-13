//! The gateway's pure decision core. It has no side effects: callers pass the clock and the
//! randomness in. The crate depends only on serde and serde_json, so purity is a property of the
//! build and does not depend on review.

mod request;
mod time;

pub use request::{
    Budgets, DEFAULT_BUDGETS, DEFAULT_MAX_BODY, DEFAULT_MAX_OUT, DataClass, Headers, Priority,
    Reject, RejectReason, Request, RunId, inspect, parse_data_class, parse_priority,
    parse_request_id,
};
pub use time::Timestamp;
