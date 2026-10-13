use ariadne_decide::{
    DEFAULT_BUDGETS, DEFAULT_MAX_BODY, DEFAULT_POLICY, DataClass, Headers, MAX_OUT_CEILING,
    Priority, Reason, Reject, RejectReason, Request, RunId, RunUsage, Snap, Verdict, WorkerView,
    estimate_tokens, inspect, parse_data_class, parse_priority, parse_request_id, should_shed,
};

use crate::{AMPLE_TENANT, DOCTOR_BODY, FIXED_NOW, after};

fn inspect_body(body: &str) -> Result<Request, Reject> {
    inspect(
        body.as_bytes(),
        Headers::default(),
        FIXED_NOW,
        DEFAULT_BUDGETS,
        DEFAULT_MAX_BODY,
    )
}

fn chat(extra: &str) -> String {
    format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#)
}

#[test]
fn parse_request_id_cases() {
    let cases = [
        (
            "doctor run",
            "crashloop-a1b2c3d4-s3",
            Some(("crashloop-a1b2c3d4", 3)),
        ),
        (
            "doctor run two digit step",
            "dx-oom-12ab34cd-s12",
            Some(("dx-oom-12ab34cd", 12)),
        ),
        ("no suffix", "crashloop-a1b2c3d4", None),
        ("empty run part", "-s3", None),
        ("s with no digits", "foo-s", None),
        ("empty string", "", None),
        ("trailing dash s but no digits at all", "foo-s-bar", None),
        ("rightmost -s wins", "a-s1-s2", Some(("a-s1", 2))),
    ];
    for (name, id, want) in cases {
        let got = parse_request_id(id);
        let got = got.as_ref().map(|(run, n)| (run.as_str(), *n));
        assert_eq!(got, want, "{name}: parse_request_id({id:?})");
    }
}

#[test]
fn parse_priority_cases() {
    let cases = [
        ("interactive", Priority::Interactive),
        ("Interactive", Priority::Interactive),
        ("INTERACTIVE", Priority::Interactive),
        ("  interactive  ", Priority::Interactive),
        ("batch", Priority::Batch),
        ("", Priority::Batch),
        ("garbage", Priority::Batch),
        ("interactive ", Priority::Interactive),
    ];
    for (input, want) in cases {
        assert_eq!(parse_priority(input), want, "parse_priority({input:?})");
    }
}

#[test]
fn parse_data_class_cases() {
    let cases = [
        ("internal", DataClass::Internal),
        ("Internal", DataClass::Internal),
        ("INTERNAL", DataClass::Internal),
        ("public", DataClass::Public),
        (" Public ", DataClass::Public),
        ("restricted", DataClass::Restricted),
        ("", DataClass::Restricted),
        ("garbage", DataClass::Restricted),
    ];
    for (input, want) in cases {
        assert_eq!(parse_data_class(input), want, "parse_data_class({input:?})");
    }
}

#[test]
fn priority_labels() {
    assert_eq!(Priority::Batch.as_str(), "batch");
    assert_eq!(Priority::Interactive.as_str(), "interactive");
}

#[test]
fn data_class_labels() {
    assert_eq!(DataClass::Restricted.as_str(), "restricted");
    assert_eq!(DataClass::Internal.as_str(), "internal");
    assert_eq!(DataClass::Public.as_str(), "public");
}

#[test]
fn safe_defaults() {
    assert_eq!(Priority::default(), Priority::Batch);
    assert_eq!(DataClass::default(), DataClass::Restricted);
}

#[test]
fn inspect_happy_path() {
    let body = DOCTOR_BODY.as_bytes();
    let headers = Headers {
        request_id: "dx-oom-12ab34cd-s3",
        tenant: "",
        app: "doctor",
        priority: "interactive",
        data_class: "restricted",
    };

    let got = inspect(body, headers, FIXED_NOW, DEFAULT_BUDGETS, DEFAULT_MAX_BODY)
        .expect("a valid doctor request passes the guard");

    assert_eq!(got.id, "dx-oom-12ab34cd-s3");
    assert_eq!(got.run, RunId::new("dx-oom-12ab34cd"));
    assert_eq!(got.step, 3);
    assert_eq!(got.tenant, "platform");
    assert_eq!(got.app, "doctor");
    assert_eq!(got.priority, Priority::Interactive);
    assert_eq!(got.class, DataClass::Restricted);
    assert_eq!(got.body_bytes, body.len());
    assert_eq!(got.max_out, 768);
    assert_eq!(got.arrived, FIXED_NOW);
    assert_eq!(got.deadline, after(10));
}

#[test]
fn inspect_batch_deadline() {
    let body = br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
    let headers = Headers {
        request_id: "run-abcd1234-s1",
        ..Headers::default()
    };

    let got = inspect(body, headers, FIXED_NOW, DEFAULT_BUDGETS, DEFAULT_MAX_BODY)
        .expect("a valid batch request passes the guard");

    assert_eq!(got.priority, Priority::Batch);
    assert_eq!(got.deadline, after(30));
}

#[test]
fn inspect_reject_reasons() {
    const MSG: &str = r#"{"role":"user","content":"hi"}"#;
    let with_messages = |rest: &str| format!(r#"{{"model":"m","messages":[{MSG}]{rest}}}"#);

    let cases = [
        (
            "body too large",
            with_messages(""),
            10,
            RejectReason::BodyTooLarge,
        ),
        (
            "not json",
            "not json at all".to_owned(),
            DEFAULT_MAX_BODY,
            RejectReason::BadJson,
        ),
        (
            "json but not an object",
            "[1, 2, 3]".to_owned(),
            DEFAULT_MAX_BODY,
            RejectReason::BadJson,
        ),
        (
            "messages missing",
            r#"{"model":"m"}"#.to_owned(),
            DEFAULT_MAX_BODY,
            RejectReason::NoMessages,
        ),
        (
            "messages empty",
            r#"{"model":"m","messages":[]}"#.to_owned(),
            DEFAULT_MAX_BODY,
            RejectReason::NoMessages,
        ),
        (
            "model missing",
            format!(r#"{{"messages":[{MSG}]}}"#),
            DEFAULT_MAX_BODY,
            RejectReason::NoModel,
        ),
        (
            "model empty",
            format!(r#"{{"model":"","messages":[{MSG}]}}"#),
            DEFAULT_MAX_BODY,
            RejectReason::NoModel,
        ),
        (
            "stream true",
            with_messages(r#","stream":true"#),
            DEFAULT_MAX_BODY,
            RejectReason::StreamUnsupported,
        ),
        (
            "client sets kv_transfer_params",
            with_messages(
                r#","kv_transfer_params":{"do_remote_prefill":true,"remote_bootstrap_addr":"http://attacker:8998"}"#,
            ),
            DEFAULT_MAX_BODY,
            RejectReason::KvTransferParams,
        ),
        (
            "max_tokens zero",
            with_messages(r#","max_tokens":0"#),
            DEFAULT_MAX_BODY,
            RejectReason::BadMaxTokens,
        ),
        (
            "max_tokens negative",
            with_messages(r#","max_tokens":-5"#),
            DEFAULT_MAX_BODY,
            RejectReason::BadMaxTokens,
        ),
        (
            "max_tokens fractional",
            with_messages(r#","max_tokens":12.5"#),
            DEFAULT_MAX_BODY,
            RejectReason::BadMaxTokens,
        ),
        (
            "max_completion_tokens not a number",
            with_messages(r#","max_completion_tokens":"five""#),
            DEFAULT_MAX_BODY,
            RejectReason::BadMaxTokens,
        ),
    ];

    assert_eq!(Reject::CODE, 400);
    for (name, body, max_body, want) in cases {
        let got = inspect(
            body.as_bytes(),
            Headers::default(),
            FIXED_NOW,
            DEFAULT_BUDGETS,
            max_body,
        )
        .expect_err(name);
        assert_eq!(got.reason, want, "{name}");
        assert!(!got.detail.is_empty(), "{name}: detail is empty");
    }
}

#[test]
fn reject_reason_codes() {
    let cases = [
        (RejectReason::BodyTooLarge, "body_too_large"),
        (RejectReason::BadJson, "bad_json"),
        (RejectReason::NoMessages, "no_messages"),
        (RejectReason::NoModel, "no_model"),
        (RejectReason::StreamUnsupported, "stream_unsupported"),
        (RejectReason::KvTransferParams, "kv_transfer_params"),
        (RejectReason::BadMaxTokens, "bad_max_tokens"),
    ];
    for (reason, want) in cases {
        assert_eq!(reason.as_str(), want);
    }
}

#[test]
fn inspect_tenant_sanitizing() {
    let body = br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
    let long_raw = "a".repeat(100);
    let want_long = "a".repeat(64);

    let cases = [
        ("empty defaults to platform", "", "platform"),
        ("whitespace only defaults to platform", "   ", "platform"),
        ("mixed case lowered", "ACME-Corp", "acme-corp"),
        ("allowed punctuation kept", "tenant.42_ok", "tenant.42_ok"),
        ("disallowed chars stripped", "Tenant!!!", "tenant"),
        ("nothing survives defaults to platform", "!!!", "platform"),
        (
            "truncated to 64 chars",
            long_raw.as_str(),
            want_long.as_str(),
        ),
    ];
    for (name, raw, want) in cases {
        let headers = Headers {
            tenant: raw,
            ..Headers::default()
        };
        let got = inspect(body, headers, FIXED_NOW, DEFAULT_BUDGETS, DEFAULT_MAX_BODY)
            .expect("a valid request passes the guard");
        assert_eq!(got.tenant, want, "{name}");
    }
}

#[test]
fn inspect_refuses_every_stream_value_a_worker_could_read_as_true() {
    let refused = [
        "true",
        "1",
        "0",
        "1.0",
        r#""true""#,
        r#""yes""#,
        r#""on""#,
        r#""false""#,
        r#""""#,
        "[]",
        "{}",
    ];
    for value in refused {
        let got = inspect_body(&chat(&format!(r#","stream":{value}"#))).expect_err(value);
        assert_eq!(
            got.reason,
            RejectReason::StreamUnsupported,
            "stream: {value}"
        );
        assert_eq!(
            got.detail,
            "stream is not supported; the gateway serves non-streaming only"
        );
    }
    for value in ["false", "null"] {
        assert!(
            inspect_body(&chat(&format!(r#","stream":{value}"#))).is_ok(),
            "stream: {value}"
        );
    }
    assert!(inspect_body(&chat("")).is_ok(), "stream absent");
}

#[test]
fn inspect_caps_max_out_and_the_tenant_gate_refuses_it() {
    let cases = [
        (r#","max_completion_tokens":10000000"#, 10_000_000),
        (r#","max_tokens":4294967296"#, MAX_OUT_CEILING),
        (r#","max_tokens":4294967297"#, MAX_OUT_CEILING),
        (
            r#","max_completion_tokens":18446744073709551615"#,
            MAX_OUT_CEILING,
        ),
        (r#","max_completion_tokens":1e30"#, MAX_OUT_CEILING),
        (r#","max_completion_tokens":1e308"#, MAX_OUT_CEILING),
    ];
    assert_eq!(MAX_OUT_CEILING, 1 << 32);
    let worker = WorkerView {
        kv_pool_tokens: 10000,
        kv_usage: 0.1,
        max_inflight: 16,
        ..WorkerView::default()
    };
    for (extra, want_max_out) in cases {
        let req = inspect_body(&chat(extra)).expect(extra);
        assert_eq!(req.max_out, want_max_out, "{extra}");

        let est_tokens = estimate_tokens(&req, None);
        assert!(est_tokens > want_max_out, "{extra}: estimate {est_tokens}");
        let snap = Snap {
            now: FIXED_NOW,
            tenant: AMPLE_TENANT,
            worker: &worker,
            resident: false,
            est_tokens,
            policy: DEFAULT_POLICY,
        };
        let Verdict::Shed(refusal) = should_shed(&req, &snap) else {
            panic!("{extra}: admitted");
        };
        assert_eq!(
            (refusal.code(), refusal.reason),
            (429, Reason::TenantTokens),
            "{extra}"
        );
    }
}

#[test]
fn estimate_tokens_saturates_on_absurd_usage() {
    let req = Request {
        body_bytes: usize::MAX,
        ..inspect_body(&chat(r#","max_tokens":1e30"#)).expect("passes the guard")
    };
    let prior = RunUsage {
        prompt_tokens: u64::MAX,
        completion_tokens: u64::MAX,
        body_bytes: 0,
    };
    assert_eq!(estimate_tokens(&req, Some(&prior)), u64::MAX);
}

#[test]
fn inspect_sanitizes_app_like_tenant() {
    let long_raw = "A".repeat(100);
    let want_long = "a".repeat(64);
    let cases = [
        ("plain", "doctor", "doctor"),
        ("mixed case lowered", " Doctor-V2 ", "doctor-v2"),
        ("label injection stripped", "doctor\",pod=\"x", "doctorpodx"),
        ("newline stripped", "a\nb", "ab"),
        ("empty stays empty", "", ""),
        ("nothing survives stays empty", "!!!", ""),
        (
            "truncated to 64 chars",
            long_raw.as_str(),
            want_long.as_str(),
        ),
    ];
    for (name, raw, want) in cases {
        let headers = Headers {
            app: raw,
            ..Headers::default()
        };
        let got = inspect(
            chat("").as_bytes(),
            headers,
            FIXED_NOW,
            DEFAULT_BUDGETS,
            DEFAULT_MAX_BODY,
        )
        .expect("a valid request passes the guard");
        assert_eq!(got.app, want, "{name}");
        assert_eq!(got.tenant, "platform", "{name}");
    }
}
