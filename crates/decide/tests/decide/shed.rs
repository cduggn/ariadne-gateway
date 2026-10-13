use std::time::Duration;

use ariadne_decide::{
    DEFAULT_POLICY, Priority, Reason, Refusal, Request, RunUsage, Snap, TenantState, Timestamp,
    Verdict, WorkerView, estimate_tokens, should_shed,
};

use crate::{AMPLE_TENANT, FIXED_NOW, after, request};

/// Checks a refusal against the fields the gate contract promises. `detail` is human-readable
/// and not part of that contract, so it is only checked for presence.
#[track_caller]
fn assert_shed(name: &str, got: Verdict, code: u16, reason: Reason, retry_after: Duration) {
    let Verdict::Shed(refusal) = got else {
        panic!("{name}: should_shed() = Admit, want {reason:?}");
    };
    assert_eq!(
        (refusal.code(), refusal.reason, refusal.retry_after),
        (code, reason, retry_after),
        "{name}"
    );
    assert!(!refusal.detail.is_empty(), "{name}: detail is empty");
}

fn snap(worker: &WorkerView) -> Snap<'_> {
    Snap {
        now: FIXED_NOW,
        tenant: AMPLE_TENANT,
        worker,
        resident: false,
        est_tokens: 0,
        policy: DEFAULT_POLICY,
    }
}

#[test]
fn admit_on_healthy_worker() {
    let r = request(Priority::Interactive, after(10));
    let worker = WorkerView {
        kv_pool_tokens: 10000,
        kv_usage: 0.5,
        in_flight: 5,
        max_inflight: 16,
        mean_service_s: 1.0,
        ttft_p50_s: 0.1,
        ttft_p99_s: 0.2,
        ..WorkerView::default()
    };
    let s = Snap {
        tenant: TenantState {
            available_tokens: 100000.0,
            rate_per_s: 1000.0,
        },
        est_tokens: 500,
        ..snap(&worker)
    };
    assert_eq!(should_shed(&r, &s), Verdict::Admit);
}

#[test]
fn tenant_tokens() {
    // Every case gives the worker a fully saturated KV pool, so a request that clears the tenant
    // gate would still shed on kv_free. A tenant refusal in every case therefore also proves the
    // gate order.
    let saturated = WorkerView {
        kv_pool_tokens: 10000,
        kv_usage: 1.0,
        in_flight: 0,
        max_inflight: 16,
        ..WorkerView::default()
    };
    let r = request(Priority::Batch, after(30));
    let tenant = |available_tokens, rate_per_s| TenantState {
        available_tokens,
        rate_per_s,
    };

    let cases = [
        ("retry-after arithmetic", tenant(4000.0, 1000.0), 10000, 6),
        (
            "retry-after clamps to a 1s minimum",
            tenant(1000.5, 1000000.0),
            1001,
            1,
        ),
        (
            "retry-after clamps to a 60s maximum",
            tenant(0.0, 1.0),
            100000,
            60,
        ),
        ("zero rate defaults to 60s", tenant(100.0, 0.0), 5000, 60),
        (
            "negative rate defaults to 60s",
            tenant(100.0, -5.0),
            5000,
            60,
        ),
        (
            "wins over a simultaneously full kv pool",
            tenant(100.0, 1000.0),
            5000,
            5,
        ),
    ];
    for (name, tenant, est_tokens, want_retry_s) in cases {
        let s = Snap {
            tenant,
            est_tokens,
            ..snap(&saturated)
        };
        assert_shed(
            name,
            should_shed(&r, &s),
            429,
            Reason::TenantTokens,
            Duration::from_secs(want_retry_s),
        );
    }
}

#[test]
fn kv_free() {
    let r = request(Priority::Batch, after(30));
    let worker = |kv_pool_tokens, kv_usage, reserved_tokens| WorkerView {
        kv_pool_tokens,
        kv_usage,
        reserved_tokens,
        in_flight: 0,
        max_inflight: 16,
        ..WorkerView::default()
    };

    let cases = [
        (
            "new run sheds at 0.19 free, below the line",
            worker(10000, 0.81, 0),
            false,
            100,
            true,
        ),
        (
            "resident continuation admitted at 0.19 free",
            worker(10000, 0.81, 0),
            true,
            100,
            false,
        ),
        (
            "resident continuation sheds at 0.04 free, below the floor",
            worker(10000, 0.96, 0),
            true,
            100,
            true,
        ),
        (
            "estimate above free tokens sheds even above the line",
            worker(1000, 0.5, 0),
            false,
            600,
            true,
        ),
        (
            "reserved tokens shrink free enough to shed",
            worker(1000, 0.5, 350),
            false,
            100,
            true,
        ),
        (
            "same pool with no reservation admits",
            worker(1000, 0.5, 0),
            false,
            100,
            false,
        ),
    ];
    for (name, worker, resident, est_tokens, want_shed) in cases {
        let s = Snap {
            resident,
            est_tokens,
            ..snap(&worker)
        };
        let got = should_shed(&r, &s);
        if want_shed {
            assert_shed(name, got, 503, Reason::KvFree, Duration::from_secs(2));
        } else {
            assert_eq!(got, Verdict::Admit, "{name}");
        }
    }
}

#[test]
fn timeout_queue() {
    // A generous KV pool on every case, so only the timeout_queue gate can refuse.
    let worker = |in_flight, queued, mean_service_s| WorkerView {
        kv_pool_tokens: 10000,
        kv_usage: 0.1,
        in_flight,
        max_inflight: 16,
        queued,
        mean_service_s,
        ..WorkerView::default()
    };
    let one_second_ago = Timestamp::from_epoch_offset(Duration::from_secs(999_999));

    let cases = [
        (
            "not full, deadline already passed, sheds",
            worker(1, 0, 5.0),
            one_second_ago,
            true,
        ),
        (
            "not full, deadline ahead, admits",
            worker(1, 0, 5.0),
            after(10),
            false,
        ),
        (
            "full, projected wait exceeds deadline, sheds",
            worker(16, 5, 2.0),
            after(5),
            true,
        ),
        (
            "full, projected wait within deadline, admits",
            worker(16, 5, 2.0),
            after(20),
            false,
        ),
        (
            "full with unknown service time never sheds",
            worker(16, 5, 0.0),
            after(1),
            false,
        ),
    ];
    for (name, worker, deadline, want_shed) in cases {
        let r = request(Priority::Batch, deadline);
        let got = should_shed(&r, &snap(&worker));
        if want_shed {
            assert_shed(name, got, 503, Reason::TimeoutQueue, Duration::from_secs(1));
        } else {
            assert_eq!(got, Verdict::Admit, "{name}");
        }
    }
}

#[test]
fn p99_spread() {
    use Priority::{Batch, Interactive};

    let cases = [
        (
            "sheds batch with spread and a rising queue",
            Batch,
            0.1,
            0.5,
            true,
            true,
        ),
        (
            "never sheds interactive with the same spread",
            Interactive,
            0.1,
            0.5,
            true,
            false,
        ),
        (
            "batch without a rising queue admits despite the spread",
            Batch,
            0.1,
            0.5,
            false,
            false,
        ),
        (
            "batch with unknown p50 admits despite a large p99",
            Batch,
            0.0,
            5.0,
            true,
            false,
        ),
        (
            "batch exactly at the spread ratio does not shed",
            Batch,
            0.1,
            0.4,
            true,
            false,
        ),
    ];
    for (name, priority, ttft_p50_s, ttft_p99_s, waiting_rising, want_shed) in cases {
        let worker = WorkerView {
            kv_pool_tokens: 10000,
            kv_usage: 0.1,
            in_flight: 0,
            max_inflight: 16,
            ttft_p50_s,
            ttft_p99_s,
            waiting_rising,
            ..WorkerView::default()
        };
        let r = request(priority, after(30));
        let got = should_shed(&r, &snap(&worker));
        if want_shed {
            assert_shed(name, got, 503, Reason::P99Spread, Duration::from_secs(5));
        } else {
            assert_eq!(got, Verdict::Admit, "{name}");
        }
    }
}

#[test]
fn refusal_code_and_stays() {
    let cases = [
        (Reason::TenantTokens, "tenant_tokens", 429, true),
        (Reason::KvFree, "kv_free", 503, false),
        (Reason::TimeoutQueue, "timeout_queue", 503, false),
        (Reason::P99Spread, "p99_spread", 503, false),
        (Reason::NoEligiblePod, "no_eligible_pod", 503, false),
        (Reason::QueueFull, "queue_full", 503, false),
    ];
    for (reason, label, code, stays) in cases {
        let refusal = Refusal {
            reason,
            retry_after: Duration::ZERO,
            detail: String::new(),
        };
        assert_eq!(reason.as_str(), label);
        assert_eq!(refusal.code(), code, "{label}");
        assert_eq!(refusal.stays(), stays, "{label}");
    }
}

#[test]
fn worker_view_free_tokens_and_ratio() {
    let worker = |kv_pool_tokens, reserved_tokens| WorkerView {
        kv_pool_tokens,
        kv_usage: 0.5,
        reserved_tokens,
        ..WorkerView::default()
    };
    let cases = [
        ("no reservation", worker(1000, 0), 500, 0.5),
        ("reservation reduces free", worker(1000, 200), 300, 0.3),
        (
            "reservation past scraped free clamps to zero",
            worker(1000, 900),
            0,
            0.0,
        ),
        ("empty pool reports zero ratio", worker(0, 0), 0, 0.0),
        (
            "usage rounds to the nearest token",
            worker(3, 0),
            2,
            2.0 / 3.0,
        ),
    ];
    for (name, worker, want_free, want_ratio) in cases {
        assert_eq!(worker.free_tokens(), want_free, "{name}");
        assert_eq!(worker.free_ratio(), want_ratio, "{name}");
    }
}

#[test]
fn estimate_tokens_cases() {
    let req = Request {
        body_bytes: 35000,
        max_out: 768,
        ..request(Priority::Batch, FIXED_NOW)
    };
    let prior = |body_bytes| RunUsage {
        prompt_tokens: 9000,
        completion_tokens: 60,
        body_bytes,
    };
    let cases = [
        ("no prior estimates from body bytes alone", None, 10768),
        (
            "with prior anchors to real usage plus growth",
            Some(prior(30000)),
            11257,
        ),
        (
            "shrinking body clamps growth to zero",
            Some(prior(40000)),
            9828,
        ),
    ];
    for (name, prior, want) in cases {
        assert_eq!(estimate_tokens(&req, prior.as_ref()), want, "{name}");
    }
}

#[test]
fn degenerate_telemetry_never_panics() {
    let r = request(Priority::Batch, after(30));
    let views = [
        WorkerView::default(),
        WorkerView {
            kv_pool_tokens: u64::MAX,
            kv_usage: f64::NAN,
            reserved_tokens: u64::MAX,
            in_flight: usize::MAX,
            queued: usize::MAX,
            max_inflight: 0,
            mean_service_s: f64::INFINITY,
            ttft_p50_s: f64::NAN,
            ttft_p99_s: f64::INFINITY,
            waiting_rising: true,
            ..WorkerView::default()
        },
        WorkerView {
            kv_pool_tokens: 1,
            kv_usage: -1e300,
            in_flight: 1,
            mean_service_s: 1e300,
            ..WorkerView::default()
        },
    ];
    let tenants = [
        AMPLE_TENANT,
        TenantState {
            available_tokens: f64::NEG_INFINITY,
            rate_per_s: f64::INFINITY,
        },
        TenantState {
            available_tokens: f64::NAN,
            rate_per_s: f64::NAN,
        },
    ];
    for worker in &views {
        let _ = (worker.free_tokens(), worker.free_ratio(), worker.load());
        let _ = worker.load_without(u64::MAX);
        for tenant in tenants {
            for est_tokens in [0, u64::MAX] {
                let s = Snap {
                    tenant,
                    est_tokens,
                    resident: est_tokens == 0,
                    ..snap(worker)
                };
                let _ = should_shed(&r, &s);
            }
        }
    }
}

#[test]
fn detail_text_matches_the_go_gateway() {
    let refusal = |r: &Request, s: &Snap<'_>| match should_shed(r, s) {
        Verdict::Shed(refusal) => refusal.detail,
        Verdict::Admit => panic!("admitted"),
    };
    let r = request(Priority::Batch, after(30));
    let worker = WorkerView {
        kv_pool_tokens: 10000,
        kv_usage: 0.81,
        in_flight: 16,
        queued: 1000,
        max_inflight: 16,
        mean_service_s: 1e9,
        ..WorkerView::default()
    };

    let tenant = Snap {
        tenant: TenantState {
            available_tokens: 1000.5,
            rate_per_s: f64::INFINITY,
        },
        est_tokens: 1001,
        ..snap(&worker)
    };
    assert_eq!(
        refusal(&r, &tenant),
        "need 1001 tokens, tenant has 1000.5 available at +Inf/s"
    );
    assert_eq!(
        refusal(&r, &snap(&worker)),
        "free ratio 0.19 below line 0.19999999999999996"
    );
    let resident = Snap {
        resident: true,
        ..snap(&worker)
    };
    assert_eq!(
        refusal(&r, &resident),
        "projected wait 9223372036.854776s past deadline, 16/16 in flight"
    );
}
