use crate::request::{DataClass, Request};
use crate::shed::Refusal;

/// `orch_overflow_total` label: a 503 reached the overflow decision and was kept on the box.
pub const OVERFLOW_BLOCKED: &str = "blocked_invariant";
/// `orch_overflow_total` label: the 503 may leave the box, but no overflow backend is configured.
pub const OVERFLOW_NO_BACKEND: &str = "no_backend";

/// Permission for one request to leave the box. `may_leave` is the only source: the field is
/// private and there is no constructor, `Default`, `Clone` or deserializer, so holding one
/// proves the stay-or-leave decision ran and allowed it.
///
/// ```compile_fail
/// let forged = ariadne_decide::Offboxable(());
/// ```
///
/// ```compile_fail
/// let forged: ariadne_decide::Offboxable = Default::default();
/// ```
#[derive(Debug)]
pub struct Offboxable(());

/// The stay-or-leave decision for a refused request. Only a 503 may leave, and never for a
/// restricted request.
pub fn may_leave(r: &Request, refusal: &Refusal) -> Option<Offboxable> {
    match r.class {
        DataClass::Restricted => None,
        DataClass::Internal | DataClass::Public => (!refusal.stays()).then_some(Offboxable(())),
    }
}

/// Names what the overflow decision did with a refusal when there is no overflow backend: a
/// request that may leave still gets the 503.
pub fn overflow_result(r: &Request, refusal: &Refusal) -> &'static str {
    match may_leave(r, refusal) {
        Some(_) => OVERFLOW_NO_BACKEND,
        None => OVERFLOW_BLOCKED,
    }
}
