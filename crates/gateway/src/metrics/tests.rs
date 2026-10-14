use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use ariadne_decide::{
    DataClass, PickPolicy, Priority, Reason as ShedReason, RejectReason, Request, RunId, RunUsage,
    Sticky, Timestamp,
};
use bytes::Bytes;

use super::text::go_float;
use super::*;
use crate::clock::Clock;
use crate::fleet::{Admission, Config, Outcome, Scrape, Ticket, WorkerConfig};
use crate::overflow::{Backend, Outcome as Overflowed};
use crate::serve::{Placed, Usage};

const PODS: [&str; 2] = ["vllm-0", "vllm-1"];

/// Every family the gateway exports. The dashboard test checks each name a panel reads against
/// this list, so a renamed metric fails the build instead of blanking a panel.
const METRIC_NAMES: [&str; 27] = [
    "orch_requests_total",
    "orch_shed_total",
    "orch_pick_total",
    "orch_pick_unknown_snapshot_total",
    "orch_sticky_total",
    "orch_completed_total",
    "orch_upstream_errors_total",
    "orch_prompt_tokens_total",
    "orch_completion_tokens_total",
    "orch_overflow_total",
    "orch_overflow_backend_available",
    "orch_hop_total",
    "orch_restricted_offbox_total",
    "orch_request_duration_seconds",
    "orch_replica_healthy",
    "orch_replica_warm",
    "orch_replica_phase",
    "orch_replica_saturating",
    "orch_replica_kv_free_ratio",
    "orch_replica_tokens_in_flight",
    "orch_replica_active_requests",
    "orch_replica_queue_depth",
    "orch_replica_snapshot_age_seconds",
    "orch_replica_mean_service_seconds",
    "orch_warmup_probe_seconds",
    "orch_replica_restarts_total",
    "orch_kv_free_ratio",
];

/// A Metrics over a Gate and an idle Fleet for two pods at a fixed clock, 10 s past its epoch so
/// a scrape can be dated in the past. The fleet never runs, so every phase stays Down and the
/// test sets the gate's state directly.
fn fixture() -> (Metrics, Gate, Timestamp) {
    let (clock, hand) = Clock::manual();
    hand.advance(Duration::from_secs(10));
    let pods: Vec<String> = PODS.map(str::to_owned).into();
    let gate = Gate::new(Config::defaults(pods), clock.clone(), |_| 0);
    let urls = BTreeMap::from([
        ("vllm-0".to_owned(), "http://127.0.0.1:1".to_owned()),
        ("vllm-1".to_owned(), "http://127.0.0.1:2".to_owned()),
    ]);
    let fleet = Fleet::new(
        gate.clone(),
        urls,
        WorkerConfig::defaults(Bytes::new()),
        reqwest::Client::new(),
        clock.clone(),
    );
    let now = clock.now();
    (Metrics::new(gate.clone(), fleet, "sliced", 3899), gate, now)
}

fn before(now: Timestamp, ms: u64) -> Timestamp {
    Timestamp::from_epoch_offset(now.epoch_offset() - Duration::from_millis(ms))
}

fn scrape(at: Timestamp, kv_usage: f64) -> Scrape {
    Scrape {
        at,
        kv_pool_tokens: 10000,
        kv_usage,
        ttft_p50_s: 0.0,
        ttft_p99_s: 0.0,
        waiting_rising: false,
    }
}

/// Fails for every exposition line missing from `body`, printing the family's lines next to
/// each miss.
#[track_caller]
fn want_lines(body: &str, lines: &[&str]) {
    let have: HashSet<&str> = body.lines().collect();
    let mut missing = Vec::new();
    for want in lines {
        if have.contains(want) {
            continue;
        }
        let family = want.split(['{', ' ']).next().unwrap_or(want);
        let near: Vec<&str> = body.lines().filter(|l| l.starts_with(family)).collect();
        missing.push(format!(
            "missing {want:?}\nfamily has:\n{}",
            near.join("\n")
        ));
    }
    assert!(missing.is_empty(), "{}", missing.join("\n"));
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// A request with no placement, status or timing, which `..event()` fills in below.
fn event() -> Event {
    Event {
        request_id: String::new(),
        run: None,
        step: 0,
        tenant: String::new(),
        priority: Priority::Batch,
        class: DataClass::Restricted,
        placement: None,
        status: None,
        reason: None,
        gateway: Duration::ZERO,
        queue: Duration::ZERO,
        upstream: Duration::ZERO,
        total: Duration::ZERO,
        usage: None,
        overflow: None,
        overflow_time: Duration::ZERO,
        hop: None,
        hop_time: Duration::ZERO,
        hop_fallback: false,
    }
}

fn placed(pod: &str, policy: PickPolicy, sticky: Sticky) -> Option<Placed> {
    Some(Placed {
        pod: pod.to_owned(),
        policy,
        sticky,
        unknown: false,
        est: 0,
    })
}

/// A placement that chose no worker.
fn unplaced(sticky: Sticky) -> Option<Placed> {
    placed("", PickPolicy::PrefixThenLoad, sticky)
}

fn usage(prompt: u64, cached: u64, completion: u64, finish: &str) -> Option<Usage> {
    Some(Usage {
        prompt_tokens: prompt,
        cached_tokens: cached,
        completion_tokens: completion,
        finish_reason: finish.to_owned(),
    })
}

fn shed(r: ShedReason) -> Option<Reason> {
    Some(Reason::Shed(r))
}

/// Covers every outcome `observe` distinguishes. A 200 with usage on vllm-0, three gateway
/// refusals, a 502 with no worker answer and a 500 relayed from a stale vllm-1.
fn representative() -> Vec<Event> {
    vec![
        Event {
            request_id: "run-a-s2".into(),
            priority: Priority::Interactive,
            class: DataClass::Internal,
            placement: placed("vllm-0", PickPolicy::PrefixThenLoad, Sticky::Hit),
            status: Some(200),
            gateway: ms(3),
            queue: ms(40),
            upstream: ms(1200),
            total: ms(1243),
            usage: usage(5000, 4500, 120, "tool_calls"),
            ..event()
        },
        Event {
            request_id: "run-b-s1".into(),
            placement: unplaced(Sticky::New),
            status: Some(429),
            reason: shed(ShedReason::TenantTokens),
            gateway: ms(1),
            total: ms(1),
            ..event()
        },
        Event {
            request_id: "run-c-s3".into(),
            placement: unplaced(Sticky::BrokenShed),
            status: Some(503),
            reason: shed(ShedReason::KvFree),
            overflow: Some(Overflowed::Blocked),
            gateway: ms(1),
            total: ms(1),
            ..event()
        },
        Event {
            request_id: "bad".into(),
            status: Some(400),
            reason: Some(Reason::Guard(RejectReason::BadJson)),
            gateway: ms(1),
            total: ms(1),
            ..event()
        },
        Event {
            request_id: "run-d-s1".into(),
            placement: placed("vllm-1", PickPolicy::PrefixThenLoad, Sticky::New),
            status: Some(502),
            reason: Some(Reason::UpstreamError),
            gateway: ms(2),
            upstream: ms(115000),
            total: ms(115002),
            ..event()
        },
        Event {
            placement: Some(Placed {
                pod: "vllm-1".to_owned(),
                policy: PickPolicy::LeastLoaded,
                sticky: Sticky::None,
                unknown: true,
                est: 0,
            }),
            status: Some(500),
            gateway: ms(2),
            upstream: ms(500),
            total: ms(502),
            ..event()
        },
    ]
}

#[test]
fn preregistered_series_appear_before_any_request() {
    let (m, _, _) = fixture();
    let body = m.render();
    want_lines(
        &body,
        &[
            r#"orch_shed_total{code="429",reason="tenant_tokens"} 0"#,
            r#"orch_shed_total{code="503",reason="kv_free"} 0"#,
            r#"orch_shed_total{code="503",reason="timeout_queue"} 0"#,
            r#"orch_shed_total{code="503",reason="p99_spread"} 0"#,
            r#"orch_shed_total{code="503",reason="no_eligible_pod"} 0"#,
            r#"orch_shed_total{code="503",reason="queue_full"} 0"#,
            r#"orch_overflow_total{result="blocked_invariant"} 0"#,
            r#"orch_overflow_total{result="no_backend"} 0"#,
            r#"orch_hop_total{result="hopped"} 0"#,
            r#"orch_hop_total{result="failed"} 0"#,
            r#"orch_hop_total{result="below_threshold"} 0"#,
            r#"orch_hop_total{result="recompute_cheaper"} 0"#,
            r#"orch_hop_total{result="busy"} 0"#,
            "orch_restricted_offbox_total 0",
        ],
    );
    assert!(
        !body.contains("orch_pick_total{"),
        "orch_pick_total has a series before any request:\n{body}"
    );
}

#[test]
fn observe_counts_hops_and_their_time() {
    let (m, _, _) = fixture();
    let moved = Event {
        placement: placed("vllm-1", PickPolicy::PrefixThenLoad, Sticky::BrokenLoad),
        status: Some(200),
        ..event()
    };
    let hopped = Event {
        hop: Some(hop::Outcome::Hopped),
        hop_time: ms(40),
        ..moved.clone()
    };
    let failed = Event {
        hop: Some(hop::Outcome::Failed),
        hop_time: ms(10),
        ..moved.clone()
    };
    let skipped = Event {
        hop: Some(hop::Outcome::BelowThreshold),
        ..moved
    };
    let plain = Event {
        placement: placed("vllm-0", PickPolicy::PrefixThenLoad, Sticky::None),
        status: Some(200),
        ..event()
    };
    for ev in [hopped, failed, skipped, plain] {
        m.observe(&ev);
    }
    want_lines(
        &m.render(),
        &[
            r#"orch_hop_total{result="hopped"} 1"#,
            r#"orch_hop_total{result="failed"} 1"#,
            r#"orch_hop_total{result="below_threshold"} 1"#,
            r#"orch_hop_total{result="recompute_cheaper"} 0"#,
            r#"orch_request_duration_seconds_count{stage="hop"} 2"#,
            r#"orch_request_duration_seconds_sum{stage="hop"} 0.05"#,
        ],
    );
}

#[test]
fn observe_counts_each_outcome() {
    let (m, _, _) = fixture();
    for ev in representative() {
        m.observe(&ev);
    }
    let body = m.render();
    want_lines(
        &body,
        &[
            r#"orch_requests_total{class="internal",priority="interactive"} 1"#,
            r#"orch_requests_total{class="restricted",priority="batch"} 5"#,
            r#"orch_shed_total{code="429",reason="tenant_tokens"} 1"#,
            r#"orch_shed_total{code="503",reason="kv_free"} 1"#,
            r#"orch_shed_total{code="400",reason="bad_json"} 1"#,
            r#"orch_shed_total{code="502",reason="upstream_error"} 1"#,
            r#"orch_shed_total{code="503",reason="queue_full"} 0"#,
            r#"orch_pick_total{pod="vllm-0",policy="prefix_then_load"} 1"#,
            r#"orch_pick_total{pod="vllm-1",policy="prefix_then_load"} 1"#,
            r#"orch_pick_total{pod="vllm-1",policy="least_loaded"} 1"#,
            r#"orch_pick_unknown_snapshot_total{pod="vllm-1"} 1"#,
            r#"orch_sticky_total{outcome="hit"} 1"#,
            r#"orch_sticky_total{outcome="new"} 2"#,
            r#"orch_sticky_total{outcome="broken_shed"} 1"#,
            r#"orch_sticky_total{outcome="none"} 1"#,
            r#"orch_completed_total{finish_reason="tool_calls",pod="vllm-0",status="200"} 1"#,
            r#"orch_completed_total{finish_reason="",pod="vllm-1",status="500"} 1"#,
            r#"orch_upstream_errors_total{kind="transport",pod="vllm-1"} 1"#,
            r#"orch_upstream_errors_total{kind="status",pod="vllm-1"} 1"#,
            r#"orch_prompt_tokens_total{kind="shared_hit",pod="vllm-0"} 3899"#,
            r#"orch_prompt_tokens_total{kind="run_hit",pod="vllm-0"} 601"#,
            r#"orch_prompt_tokens_total{kind="miss",pod="vllm-0"} 500"#,
            r#"orch_completion_tokens_total{pod="vllm-0"} 120"#,
            r#"orch_overflow_total{result="blocked_invariant"} 1"#,
            r#"orch_overflow_total{result="no_backend"} 0"#,
            "orch_restricted_offbox_total 0",
            r#"orch_request_duration_seconds_count{stage="gateway"} 6"#,
            r#"orch_request_duration_seconds_count{stage="e2e"} 6"#,
            r#"orch_request_duration_seconds_count{stage="queue"} 1"#,
            r#"orch_request_duration_seconds_sum{stage="queue"} 0.04"#,
            r#"orch_request_duration_seconds_bucket{stage="queue",le="0.025"} 0"#,
            r#"orch_request_duration_seconds_bucket{stage="queue",le="0.05"} 1"#,
            r#"orch_request_duration_seconds_count{stage="local"} 3"#,
            r#"orch_request_duration_seconds_bucket{stage="local",le="60"} 2"#,
            r#"orch_request_duration_seconds_bucket{stage="local",le="120"} 3"#,
        ],
    );
    for absent in [
        r#"orch_shed_total{code="500""#,
        r#"orch_completed_total{finish_reason="",pod="","#,
        r#"pod="""#,
    ] {
        assert!(
            !body.contains(absent),
            "found {absent:?}, which no outcome should produce"
        );
    }
}

fn request(now: Timestamp, id: &str, step: u64) -> Request {
    Request {
        id: id.to_owned(),
        run: RunId::new("run-m"),
        step,
        tenant: "platform".to_owned(),
        app: String::new(),
        priority: Priority::Interactive,
        class: DataClass::Restricted,
        body_bytes: 350,
        max_out: 100,
        arrived: now,
        deadline: now.saturating_add(Duration::from_secs(10)),
    }
}

/// Scrapes vllm-0 1.5 s ago at 30% KV and vllm-1 4 s ago at 90%, readies vllm-0, and leaves a
/// 300-token request in flight on it. The returned ticket keeps it in flight.
async fn load_fleet(gate: &Gate, now: Timestamp) -> Ticket {
    gate.observe("vllm-0", scrape(before(now, 1500), 0.3));
    gate.observe("vllm-1", scrape(before(now, 4000), 0.9));
    gate.set_ready("vllm-0", true);

    let Admission::Admitted(d, first) = gate.admit(&request(now, "run-m-s1", 1)).await else {
        panic!("first admit refused");
    };
    assert_eq!(d.est, 200, "first admit's estimate");
    first.settle(Outcome::Answered {
        usage: RunUsage {
            prompt_tokens: 150,
            completion_tokens: 50,
            body_bytes: 350,
        },
        latency: Duration::from_secs(2),
    });
    // Step 2 is estimated from step 1's settled usage, 150 + 50 plus the 100 max out, so it
    // reserves 300 tokens and leaves 6700 of 10000 free.
    let Admission::Admitted(d, second) = gate.admit(&request(now, "run-m-s2", 2)).await else {
        panic!("second admit refused");
    };
    assert_eq!(d.est, 300, "second admit's estimate");
    second
}

#[tokio::test]
async fn collector_derives_replica_gauges_from_gate_and_fleet() {
    let (m, gate, now) = fixture();
    let _in_flight = load_fleet(&gate, now).await;
    want_lines(
        &m.render(),
        &[
            r#"orch_replica_healthy{pod="vllm-0",pool="sliced"} 1"#,
            r#"orch_replica_healthy{pod="vllm-1",pool="sliced"} 0"#,
            r#"orch_replica_warm{pod="vllm-0",pool="sliced"} 0"#,
            r#"orch_replica_phase{phase="down",pod="vllm-0",pool="sliced"} 1"#,
            r#"orch_replica_phase{phase="warming",pod="vllm-0",pool="sliced"} 0"#,
            r#"orch_replica_phase{phase="ready",pod="vllm-0",pool="sliced"} 0"#,
            r#"orch_replica_saturating{pod="vllm-0",pool="sliced"} 0"#,
            r#"orch_replica_saturating{pod="vllm-1",pool="sliced"} 1"#,
            r#"orch_replica_kv_free_ratio{pod="vllm-0",pool="sliced"} 0.67"#,
            r#"orch_replica_kv_free_ratio{pod="vllm-1",pool="sliced"} 0.1"#,
            r#"orch_replica_tokens_in_flight{pod="vllm-0",pool="sliced"} 300"#,
            r#"orch_replica_tokens_in_flight{pod="vllm-1",pool="sliced"} 0"#,
            r#"orch_replica_active_requests{pod="vllm-0",pool="sliced"} 1"#,
            r#"orch_replica_queue_depth{pod="vllm-0",pool="sliced"} 0"#,
            r#"orch_replica_snapshot_age_seconds{pod="vllm-0",pool="sliced"} 1.5"#,
            r#"orch_replica_snapshot_age_seconds{pod="vllm-1",pool="sliced"} 4"#,
            r#"orch_replica_mean_service_seconds{pod="vllm-0",pool="sliced"} 2"#,
            r#"orch_replica_mean_service_seconds{pod="vllm-1",pool="sliced"} 0"#,
            r#"orch_warmup_probe_seconds{pod="vllm-0",pool="sliced"} 0"#,
            r#"orch_replica_restarts_total{pod="vllm-0",pool="sliced"} 0"#,
            r#"orch_kv_free_ratio{pool="sliced"} 0.67"#,
        ],
    );
}

#[test]
fn fleet_kv_free_ratio_is_zero_with_no_ready_pod() {
    let (m, gate, now) = fixture();
    gate.observe("vllm-0", scrape(now, 0.3));
    want_lines(&m.render(), &[r#"orch_kv_free_ratio{pool="sliced"} 0"#]);
}

/// Every "expr" and "query" string in a dashboard's JSON tree, through rows and nested panels.
fn exprs_of<'a>(v: &'a serde_json::Value, out: &mut Vec<&'a str>) {
    match v {
        serde_json::Value::Object(node) => {
            for (k, child) in node {
                match child.as_str() {
                    Some(s) if k == "expr" || k == "query" => out.push(s),
                    _ => exprs_of(child, out),
                }
            }
        }
        serde_json::Value::Array(items) => items.iter().for_each(|c| exprs_of(c, out)),
        _ => {}
    }
}

/// Every `orch_[a-z_]+` token in `expr`.
fn orch_names(expr: &str) -> Vec<&str> {
    let mut names = Vec::new();
    let mut rest = expr;
    while let Some(at) = rest.find("orch_") {
        let tail = &rest[at..];
        let len = tail
            .find(|c: char| !(c.is_ascii_lowercase() || c == '_'))
            .unwrap_or(tail.len());
        names.push(&tail[..len]);
        rest = &tail[len..];
    }
    names
}

#[tokio::test]
async fn dashboard_reads_only_metrics_the_gateway_exports() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../deploy/dashboards/gateway.json"
    );
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let dash: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("gateway.json is not JSON: {e}"));
    assert_eq!(dash["uid"], "cd-gateway", "dashboard uid");

    let mut exprs = Vec::new();
    exprs_of(&dash, &mut exprs);
    assert!(
        exprs.len() >= 14,
        "found {} exprs, want at least one per panel",
        exprs.len()
    );
    let mut used = HashSet::new();
    for e in &exprs {
        for token in orch_names(e) {
            let name = ["_bucket", "_sum", "_count"]
                .iter()
                .fold(token, |n, suffix| n.strip_suffix(suffix).unwrap_or(n));
            used.insert(name);
            assert!(
                METRIC_NAMES.contains(&name),
                "dashboard reads {name}, which the gateway does not export (expr {e:?})"
            );
        }
    }
    for name in [
        "orch_requests_total",
        "orch_shed_total",
        "orch_replica_kv_free_ratio",
        "orch_restricted_offbox_total",
    ] {
        assert!(used.contains(name), "dashboard never reads {name}");
    }

    let (m, gate, now) = fixture();
    gate.observe("vllm-0", scrape(now, 0.3));
    for ev in representative() {
        m.observe(&ev);
    }
    let body = m.render();
    for name in METRIC_NAMES {
        assert!(
            body.contains(&format!("# TYPE {name} ")),
            "METRIC_NAMES lists {name} but /metrics does not export it"
        );
    }
}

#[test]
fn overflow_series_are_preregistered_and_the_backend_reads_unavailable() {
    let (m, _, _) = fixture();
    want_lines(
        &m.render(),
        &[
            r#"orch_overflow_total{result="blocked_invariant"} 0"#,
            r#"orch_overflow_total{result="no_backend"} 0"#,
            r#"orch_overflow_total{result="sent"} 0"#,
            r#"orch_overflow_total{result="failed"} 0"#,
            r#"orch_overflow_total{result="unavailable"} 0"#,
            r#"orch_overflow_total{result="capped"} 0"#,
            "orch_overflow_backend_available 0",
            "orch_restricted_offbox_total 0",
        ],
    );
}

#[test]
fn observe_counts_overflow_outcomes_and_their_time() {
    let (m, _, _) = fixture();
    let refused = Event {
        status: Some(503),
        reason: shed(ShedReason::KvFree),
        class: DataClass::Internal,
        placement: unplaced(Sticky::New),
        gateway: ms(1),
        total: ms(1),
        ..event()
    };
    let sent = Event {
        status: Some(200),
        class: DataClass::Internal,
        placement: unplaced(Sticky::New),
        overflow: Some(Overflowed::Sent),
        overflow_time: ms(800),
        gateway: ms(1),
        total: ms(801),
        usage: usage(0, 0, 30, "tool_calls"),
        ..event()
    };
    let failed = Event {
        overflow: Some(Overflowed::Failed),
        overflow_time: ms(200),
        ..refused.clone()
    };
    let unavailable = Event {
        overflow: Some(Overflowed::Unavailable),
        ..refused.clone()
    };
    let capped = Event {
        overflow: Some(Overflowed::Capped),
        ..refused
    };
    for ev in [sent, failed, unavailable, capped] {
        m.observe(&ev);
    }
    let body = m.render();
    want_lines(
        &body,
        &[
            r#"orch_overflow_total{result="sent"} 1"#,
            r#"orch_overflow_total{result="failed"} 1"#,
            r#"orch_overflow_total{result="unavailable"} 1"#,
            r#"orch_overflow_total{result="capped"} 1"#,
            r#"orch_overflow_total{result="blocked_invariant"} 0"#,
            r#"orch_shed_total{code="503",reason="kv_free"} 3"#,
            r#"orch_completed_total{finish_reason="tool_calls",pod="overflow",status="200"} 1"#,
            r#"orch_request_duration_seconds_count{stage="overflow"} 2"#,
            r#"orch_request_duration_seconds_sum{stage="overflow"} 1"#,
            "orch_restricted_offbox_total 0",
        ],
    );
    for absent in [
        r#"pod="""#,
        r#"orch_pick_total{pod="overflow""#,
        r#"orch_prompt_tokens_total{kind="miss",pod="overflow""#,
    ] {
        assert!(
            !body.contains(absent),
            "found {absent:?}, which an overflowed answer must not produce"
        );
    }
}

#[test]
fn a_restricted_answer_from_off_box_would_count() {
    // Unreachable through the gateway (may_leave, the permit, the backend's own check), but if
    // it ever happened the alert must fire.
    let (m, _, _) = fixture();
    m.observe(&Event {
        status: Some(200),
        class: DataClass::Restricted,
        overflow: Some(Overflowed::Sent),
        overflow_time: ms(5),
        ..event()
    });
    want_lines(&m.render(), &["orch_restricted_offbox_total 1"]);
}

#[tokio::test]
async fn watch_overflow_follows_the_circuit() {
    let (m, _, _) = fixture();
    let ov = Arc::new(Overflow::new(
        Backend::Null,
        Duration::ZERO,
        1,
        Arc::new(|| Duration::ZERO),
        Arc::new(|_, _, _| {}),
    ));
    m.watch_overflow(Arc::clone(&ov));
    ov.probe().await;
    want_lines(&m.render(), &["orch_overflow_backend_available 0"]);
}

/// The sequence the Go gateway's metrics package was fed to record `go_exposition.txt`: the
/// representative events, then hops, every overflow outcome, a restricted answer from off the
/// box, a client that went away, token counts past 1e6, a queue refusal and a finish reason
/// that needs escaping.
fn golden_sequence() -> Vec<Event> {
    let mut evs = representative();
    evs.extend([
        Event {
            placement: placed("vllm-1", PickPolicy::PrefixThenLoad, Sticky::BrokenLoad),
            priority: Priority::Interactive,
            class: DataClass::Internal,
            status: Some(200),
            hop: Some(hop::Outcome::Hopped),
            hop_time: ms(40),
            gateway: ms(2),
            upstream: ms(300),
            total: ms(342),
            usage: usage(2000, 100, 10, "stop"),
            ..event()
        },
        Event {
            placement: placed("vllm-1", PickPolicy::PrefixThenLoad, Sticky::BrokenLoad),
            status: Some(200),
            hop: Some(hop::Outcome::Failed),
            hop_time: ms(10),
            gateway: ms(1),
            total: ms(11),
            ..event()
        },
        Event {
            placement: placed("vllm-0", PickPolicy::P2c, Sticky::BrokenLoad),
            status: Some(200),
            hop: Some(hop::Outcome::BelowThreshold),
            ..event()
        },
        Event {
            status: Some(200),
            class: DataClass::Internal,
            placement: unplaced(Sticky::New),
            overflow: Some(Overflowed::Sent),
            overflow_time: ms(800),
            gateway: ms(1),
            total: ms(801),
            usage: usage(40, 0, 30, "tool_calls"),
            ..event()
        },
        Event {
            status: Some(503),
            reason: shed(ShedReason::KvFree),
            class: DataClass::Internal,
            placement: unplaced(Sticky::New),
            overflow: Some(Overflowed::Failed),
            overflow_time: ms(200),
            gateway: ms(1),
            total: ms(201),
            ..event()
        },
        Event {
            status: Some(503),
            reason: shed(ShedReason::KvFree),
            class: DataClass::Public,
            placement: unplaced(Sticky::New),
            overflow: Some(Overflowed::Unavailable),
            gateway: ms(1),
            total: ms(1),
            ..event()
        },
        Event {
            status: Some(503),
            reason: shed(ShedReason::TimeoutQueue),
            class: DataClass::Internal,
            placement: unplaced(Sticky::New),
            overflow: Some(Overflowed::Capped),
            gateway: ms(1),
            total: ms(1),
            ..event()
        },
        Event {
            status: Some(503),
            reason: shed(ShedReason::P99Spread),
            class: DataClass::Internal,
            placement: unplaced(Sticky::New),
            overflow: Some(Overflowed::Quota),
            gateway: ms(1),
            total: ms(1),
            ..event()
        },
        Event {
            status: Some(200),
            class: DataClass::Restricted,
            overflow: Some(Overflowed::Sent),
            overflow_time: ms(5),
            ..event()
        },
        Event {
            placement: placed("vllm-0", PickPolicy::PrefixThenLoad, Sticky::Hit),
            reason: Some(Reason::ClientGone),
            gateway: ms(1),
            queue: ms(2),
            upstream: Duration::from_micros(7),
            total: ms(3),
            ..event()
        },
        Event {
            placement: placed("vllm-0", PickPolicy::PrefixThenLoad, Sticky::Hit),
            priority: Priority::Interactive,
            class: DataClass::Public,
            status: Some(200),
            gateway: Duration::from_micros(10),
            upstream: Duration::from_secs(150),
            total: Duration::from_secs(150),
            usage: usage(5_000_000, 4_000_000, 1_234_567, "length"),
            ..event()
        },
        Event {
            placement: unplaced(Sticky::None),
            status: Some(503),
            reason: shed(ShedReason::QueueFull),
            gateway: ms(1),
            queue: ms(2),
            total: ms(3),
            ..event()
        },
        Event {
            placement: placed("vllm-1", PickPolicy::LeastLoaded, Sticky::BrokenGone),
            status: Some(200),
            usage: usage(100, 3000, 1, "we\"ird\\\nreason"),
            gateway: ms(1),
            upstream: ms(5),
            total: ms(6),
            ..event()
        },
    ]);
    evs
}

#[tokio::test]
async fn render_matches_the_go_gateway_byte_for_byte() {
    let (m, gate, now) = fixture();
    let _in_flight = load_fleet(&gate, now).await;
    for ev in golden_sequence() {
        m.observe(&ev);
    }
    let got = m.render();
    let want = include_str!("testdata/go_exposition.txt");
    for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
        assert_eq!(g, w, "line {}", i + 1);
    }
    assert_eq!(got, want);
}

#[test]
fn go_float_matches_strconv() {
    for (v, want) in [
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (-1.0, "-1"),
        (0.005, "0.005"),
        (1e-05, "1e-05"),
        (0.0001, "0.0001"),
        (0.00012345, "0.00012345"),
        (123456.0, "123456"),
        (999999.0, "999999"),
        (1e6, "1e+06"),
        (1234567.0, "1.234567e+06"),
        (3.899e6, "3.899e+06"),
        (1e20, "1e+20"),
        (1e21, "1e+21"),
        (1.0 / 3.0, "0.3333333333333333"),
        (120.0, "120"),
        (2.5, "2.5"),
        (-2.5e-7, "-2.5e-07"),
        (5e-324, "5e-324"),
        (f64::MAX, "1.7976931348623157e+308"),
        (f64::INFINITY, "+Inf"),
        (f64::NEG_INFINITY, "-Inf"),
        (f64::NAN, "NaN"),
    ] {
        assert_eq!(go_float(v), want, "go_float({v:e})");
    }
}
