use std::time::Duration;

use ariadne_decide::{
    DEFAULT_BUDGETS, DEFAULT_MAX_BODY, DataClass, Headers, OVERFLOW_BLOCKED, OVERFLOW_NO_BACKEND,
    Priority, Reason, Refusal, Request, inspect, may_leave, overflow_result, parse_data_class,
};
use proptest::prelude::*;

use crate::{DOCTOR_BODY, FIXED_NOW, request};

fn shed(reason: Reason) -> Refusal {
    Refusal {
        reason,
        retry_after: Duration::ZERO,
        detail: String::new(),
    }
}

#[test]
fn may_leave_cases() {
    let cases = [
        (
            "restricted 503 stays",
            DataClass::Restricted,
            Reason::KvFree,
            false,
            OVERFLOW_BLOCKED,
        ),
        (
            "internal 503 may leave",
            DataClass::Internal,
            Reason::KvFree,
            true,
            OVERFLOW_NO_BACKEND,
        ),
        (
            "public 503 may leave",
            DataClass::Public,
            Reason::QueueFull,
            true,
            OVERFLOW_NO_BACKEND,
        ),
        (
            "public 429 stays",
            DataClass::Public,
            Reason::TenantTokens,
            false,
            OVERFLOW_BLOCKED,
        ),
    ];
    for (name, class, reason, want_permit, want_label) in cases {
        let r = Request {
            id: "run-s1".to_owned(),
            class,
            ..request(Priority::Batch, FIXED_NOW)
        };
        let refusal = shed(reason);
        assert_eq!(may_leave(&r, &refusal).is_some(), want_permit, "{name}");
        assert_eq!(overflow_result(&r, &refusal), want_label, "{name}");
    }
}

#[test]
fn overflow_labels() {
    assert_eq!(OVERFLOW_BLOCKED, "blocked_invariant");
    assert_eq!(OVERFLOW_NO_BACKEND, "no_backend");
}

/// Drives a header and body through `inspect` and a refusal through `may_leave`. A request that
/// parsed as restricted must never come out with a permit.
fn check_restricted_never_leaves(
    class: &str,
    body: &str,
    reason: Reason,
) -> Result<(), TestCaseError> {
    let headers = Headers {
        request_id: "run-s1",
        data_class: class,
        ..Headers::default()
    };
    let Ok(req) = inspect(
        body.as_bytes(),
        headers,
        FIXED_NOW,
        DEFAULT_BUDGETS,
        DEFAULT_MAX_BODY,
    ) else {
        return Ok(());
    };
    let permit = may_leave(&req, &shed(reason));
    let restricted = parse_data_class(class) == DataClass::Restricted;

    prop_assert!(
        !(restricted && permit.is_some()),
        "restricted request (X-Data-Class {class:?}) got an overflow permit"
    );
    prop_assert_eq!(req.class, parse_data_class(class));
    prop_assert!(
        restricted || reason.code() != 503 || permit.is_some(),
        "{} 503 was kept on the box",
        class.trim()
    );
    Ok(())
}

const EMPTY_MESSAGES: &str = r#"{"model":"m","messages":[]}"#;

#[test]
fn restricted_never_leaves_seeds() {
    let seeds = [
        ("restricted", DOCTOR_BODY, Reason::KvFree),
        ("", DOCTOR_BODY, Reason::KvFree),
        ("RESTRICTED", DOCTOR_BODY, Reason::KvFree),
        (" internal", DOCTOR_BODY, Reason::KvFree),
        ("public", DOCTOR_BODY, Reason::KvFree),
        ("internal", EMPTY_MESSAGES, Reason::KvFree),
        ("public", DOCTOR_BODY, Reason::TenantTokens),
    ];
    for (class, body, reason) in seeds {
        check_restricted_never_leaves(class, body, reason).unwrap();
    }
}

fn any_reason() -> impl Strategy<Value = Reason> {
    prop::sample::select(vec![
        Reason::TenantTokens,
        Reason::KvFree,
        Reason::TimeoutQueue,
        Reason::P99Spread,
        Reason::NoEligiblePod,
        Reason::QueueFull,
    ])
}

/// Mostly near-misses of the three class names (case, padding, a stray character), because an
/// arbitrary string almost always parses as restricted and would leave the leave path untested.
fn any_class() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => r"[ \t]{0,2}(?i:restricted|internal|public)[ \t\x{a0}]{0,2}",
        1 => r"(?i:restricted|internal|public).",
        1 => r".(?i:restricted|internal|public)",
        1 => any::<String>(),
    ]
}

/// Mostly bodies that pass the guard, for the same reason.
fn any_body() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => Just(DOCTOR_BODY.to_owned()),
        1 => Just(EMPTY_MESSAGES.to_owned()),
        1 => any::<String>(),
    ]
}

proptest! {
    #[test]
    fn restricted_never_leaves(class in any_class(), body in any_body(), reason in any_reason()) {
        check_restricted_never_leaves(&class, &body, reason)?;
    }
}
