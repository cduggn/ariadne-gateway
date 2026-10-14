//! Moves a run's KV cache to the worker the gateway is moving the run to, so
//! that worker reads the run's history instead of recomputing it. Both
//! workers are the fleet's own pods, so a hop never sends data off the box.

pub mod hopper;
pub mod mooncake;
pub mod policy;

pub use hopper::{Clock, Hopper, Move, MoveError, NewError, Report};
pub use mooncake::{
    DEFAULT_BOOTSTRAP_PORT, Endpoint, HopError, Mooncake, Params, bootstrap_url, receive,
};
pub use policy::{
    Config, ConfigError, DEFAULT_CONFIG, Estimate, Invalid, Outcome, Plan, Skip, moved,
};
