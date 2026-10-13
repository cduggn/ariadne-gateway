use std::time::Duration;

use ariadne_decide::{
    DEFAULT_POLICY, DEFAULT_STALENESS, DEFAULT_STICK_SLACK, Fleet, PickPolicy, Placement, Priority,
    Reason, Request, RunId, Sticky, TenantState, WorkerState, WorkerView, parse_pick_policy, pick,
};

use crate::{AMPLE_TENANT, FIXED_NOW, after, request};

/// A ready, unstale, lightly loaded worker: 0.9 KV free, nothing in flight or queued, a cap of
/// 16. Tests override only the fields that matter to them.
fn fresh_worker(pod: &str) -> WorkerState {
    WorkerState {
        ready: true,
        age: Duration::ZERO,
        view: WorkerView {
            pod: pod.to_owned(),
            kv_pool_tokens: 10000,
            kv_usage: 0.1,
            max_inflight: 16,
            mean_service_s: 1.0,
            ..WorkerView::default()
        },
    }
}

fn with_in_flight(mut w: WorkerState, n: usize) -> WorkerState {
    w.view.in_flight = n;
    w
}

fn with_kv_usage(mut w: WorkerState, usage: f64) -> WorkerState {
    w.view.kv_usage = usage;
    w
}

fn with_age(mut w: WorkerState, secs: u64) -> WorkerState {
    w.age = Duration::from_secs(secs);
    w
}

fn not_ready(mut w: WorkerState) -> WorkerState {
    w.ready = false;
    w
}

/// Sets the worker's KV pool and the tokens in use, so a test can speak in tokens.
fn with_pool(mut w: WorkerState, pool: u64, used: u64) -> WorkerState {
    w.view.kv_pool_tokens = pool;
    w.view.kv_usage = used as f64 / pool as f64;
    w
}

/// Requests with a generous deadline, so only the gate under test can shed. Stickiness comes
/// entirely from `Fleet::bound`.
fn req_run(run: &str) -> Request {
    Request {
        run: RunId::new(run),
        ..req_no_run()
    }
}

fn req_no_run() -> Request {
    request(Priority::Batch, after(30))
}

fn fleet_of(workers: impl IntoIterator<Item = WorkerState>) -> Fleet {
    Fleet {
        now: FIXED_NOW,
        workers: workers.into_iter().collect(),
        tenant: AMPLE_TENANT,
        est_tokens: 100,
        bound: None,
        history: 0,
        shed: DEFAULT_POLICY,
        staleness: DEFAULT_STALENESS,
        stick_slack: DEFAULT_STICK_SLACK,
    }
}

fn bound_to(pod: &str, mut f: Fleet) -> Fleet {
    f.bound = Some(pod.to_owned());
    f
}

fn no_rnd(n: usize) -> usize {
    panic!("rnd should not be called, got n={n}");
}

#[track_caller]
fn assert_placed(got: &Placement, pod: &str, sticky: Sticky, unknown: bool) {
    assert_eq!(got.pod(), Some(pod));
    assert_eq!(got.sticky, sticky);
    assert_eq!(got.unknown(), unknown);
    assert!(got.placed());
    assert_eq!(got.result.as_ref().err(), None);
}

#[track_caller]
fn assert_shed(got: &Placement, code: u16, reason: Reason, sticky: Sticky) {
    let refusal = got.result.as_ref().expect_err("pick placed the request");
    assert_eq!((refusal.code(), refusal.reason), (code, reason));
    assert_eq!(got.sticky, sticky);
    assert_eq!(got.pod(), None);
    assert!(!got.unknown());
    assert!(!got.placed());
}

#[test]
fn new_run_to_least_loaded() {
    let f = fleet_of([
        fresh_worker("pod-a"),
        with_in_flight(fresh_worker("pod-b"), 10),
    ]);
    let got = pick(&req_run("run1"), &f, PickPolicy::PrefixThenLoad, no_rnd);
    assert_placed(&got, "pod-a", Sticky::New, false);
    assert_eq!(got.policy, PickPolicy::PrefixThenLoad);
}

#[test]
fn sticky_hit() {
    // Loads 0.2875 and 0.1.
    let f = fleet_of([
        with_in_flight(fresh_worker("pod-a"), 3),
        fresh_worker("pod-b"),
    ]);
    let got = pick(
        &req_run("run1"),
        &bound_to("pod-a", f),
        PickPolicy::PrefixThenLoad,
        no_rnd,
    );
    assert_placed(&got, "pod-a", Sticky::Hit, false);
}

#[test]
fn sticky_broken_load_over_slack() {
    // pod-a at load 0.6 is 0.5 over pod-b.
    let f = fleet_of([
        with_in_flight(fresh_worker("pod-a"), 8),
        fresh_worker("pod-b"),
    ]);
    let got = pick(
        &req_run("run1"),
        &bound_to("pod-a", f),
        PickPolicy::PrefixThenLoad,
        no_rnd,
    );
    assert_placed(&got, "pod-b", Sticky::BrokenLoad, false);
}

#[test]
fn sticky_kept_within_slack() {
    // pod-a is exactly the slack (0.25 = 4/16) busier than pod-b, so pick keeps it.
    let f = fleet_of([
        with_in_flight(fresh_worker("pod-a"), 4),
        fresh_worker("pod-b"),
    ]);
    let got = pick(
        &req_run("run1"),
        &bound_to("pod-a", f),
        PickPolicy::PrefixThenLoad,
        no_rnd,
    );
    assert_placed(&got, "pod-a", Sticky::Hit, false);
}

#[test]
fn broken_shed_when_bound_is_kv_full() {
    // 0.03 free is below the 0.05 resident floor, so the bound worker itself refuses even though
    // it is exempt from the ordinary 0.20 line.
    let f = fleet_of([
        with_kv_usage(fresh_worker("pod-a"), 0.97),
        fresh_worker("pod-b"),
    ]);
    let got = pick(
        &req_run("run1"),
        &bound_to("pod-a", f),
        PickPolicy::PrefixThenLoad,
        no_rnd,
    );
    assert_placed(&got, "pod-b", Sticky::BrokenShed, false);
}

#[test]
fn broken_gone_when_bound_not_ready() {
    let f = fleet_of([not_ready(fresh_worker("pod-a")), fresh_worker("pod-b")]);
    let got = pick(
        &req_run("run1"),
        &bound_to("pod-a", f),
        PickPolicy::PrefixThenLoad,
        no_rnd,
    );
    assert_placed(&got, "pod-b", Sticky::BrokenGone, false);
}

#[test]
fn excludes_not_ready_and_down_workers() {
    let f = fleet_of([
        not_ready(fresh_worker("pod-a")),
        with_age(fresh_worker("pod-b"), 10),
        fresh_worker("pod-c"),
    ]);
    let got = pick(&req_no_run(), &f, PickPolicy::PrefixThenLoad, no_rnd);
    assert_placed(&got, "pod-c", Sticky::None, false);
}

#[test]
fn stale_floor_two_workers_both_stay() {
    // fresh/total = 16/32 = 0.5, below the 0.75 floor, so the stale worker stays and pick
    // chooses it as the less loaded one, on stale telemetry.
    let f = fleet_of([
        with_age(fresh_worker("pod-a"), 3),
        with_in_flight(fresh_worker("pod-b"), 8),
    ]);
    let got = pick(&req_no_run(), &f, PickPolicy::PrefixThenLoad, no_rnd);
    assert_placed(&got, "pod-a", Sticky::None, true);
}

#[test]
fn stale_floor_eight_workers_two_dropped() {
    // fresh/total = 96/128 = 0.75, at the floor, so pick drops both stale workers even though
    // they are the least loaded candidates.
    let f = fleet_of([
        with_age(fresh_worker("stale-0"), 3),
        with_age(fresh_worker("stale-1"), 3),
        with_in_flight(fresh_worker("fresh-0"), 0),
        with_in_flight(fresh_worker("fresh-1"), 4),
        with_in_flight(fresh_worker("fresh-2"), 4),
        with_in_flight(fresh_worker("fresh-3"), 4),
        with_in_flight(fresh_worker("fresh-4"), 4),
        with_in_flight(fresh_worker("fresh-5"), 4),
    ]);
    let got = pick(&req_no_run(), &f, PickPolicy::PrefixThenLoad, no_rnd);
    assert_placed(&got, "fresh-0", Sticky::None, false);
}

#[test]
fn all_down_is_no_eligible_pod() {
    let f = fleet_of([
        with_age(fresh_worker("pod-a"), 10),
        with_age(fresh_worker("pod-b"), 20),
    ]);
    let got = pick(&req_run("run1"), &f, PickPolicy::PrefixThenLoad, no_rnd);
    assert_shed(&got, 503, Reason::NoEligiblePod, Sticky::New);
    let refusal = got.result.expect_err("shed");
    assert_eq!(refusal.retry_after, Duration::from_secs(2));
    assert_eq!(
        refusal.detail,
        "2 workers, 0 ready, none survive staleness filtering"
    );
}

#[test]
fn tenant_exhausted_outranks_kv_full() {
    let mut f = fleet_of([
        with_kv_usage(fresh_worker("pod-a"), 0.97),
        fresh_worker("pod-b"),
    ]);
    f.tenant = TenantState {
        available_tokens: 1.0,
        rate_per_s: 1000.0,
    };
    let got = pick(&req_no_run(), &f, PickPolicy::PrefixThenLoad, no_rnd);
    assert_shed(&got, 429, Reason::TenantTokens, Sticky::None);
}

#[test]
fn all_kv_full_is_kv_free() {
    let f = fleet_of([
        with_kv_usage(fresh_worker("pod-a"), 0.9),
        with_kv_usage(fresh_worker("pod-b"), 0.85),
    ]);
    let got = pick(&req_no_run(), &f, PickPolicy::PrefixThenLoad, no_rnd);
    assert_shed(&got, 503, Reason::KvFree, Sticky::None);
}

#[test]
fn least_loaded_tie_broken_by_pod_name() {
    let f = fleet_of([fresh_worker("pod-b"), fresh_worker("pod-a")]);
    let got = pick(&req_no_run(), &f, PickPolicy::LeastLoaded, no_rnd);
    assert_placed(&got, "pod-a", Sticky::None, false);
}

#[test]
fn least_loaded_reports_sticky_for_metrics_only() {
    // The bound worker is admitted but busier than pod-b. least_loaded still takes pod-b, and
    // reports broken_load, not hit.
    let f = fleet_of([
        with_in_flight(fresh_worker("pod-a"), 8),
        fresh_worker("pod-b"),
    ]);
    let got = pick(
        &req_run("run1"),
        &bound_to("pod-a", f),
        PickPolicy::LeastLoaded,
        no_rnd,
    );
    assert_placed(&got, "pod-b", Sticky::BrokenLoad, false);
}

#[test]
fn p2c_samples_two_and_takes_the_less_loaded() {
    // pod-c is tied for the global minimum, but a rigged rnd samples only pod-a and pod-b, so
    // p2c must return the lower of that pair.
    let f = fleet_of([
        fresh_worker("pod-a"),
        with_in_flight(fresh_worker("pod-b"), 8),
        with_in_flight(fresh_worker("pod-c"), 0),
    ]);

    // rnd(3) -> 0 is pod-a; rnd(2) -> 0 is bumped past it to pod-b.
    let mut calls = Vec::new();
    let rnd = |n| {
        calls.push(n);
        0
    };

    let got = pick(&req_no_run(), &f, PickPolicy::P2c, rnd);
    assert_placed(&got, "pod-a", Sticky::None, false);
    assert_eq!(calls, [3, 2]);
}

#[test]
fn p2c_never_calls_rnd_for_one_candidate() {
    let f = fleet_of([fresh_worker("pod-a")]);
    let got = pick(&req_no_run(), &f, PickPolicy::P2c, no_rnd);
    assert_placed(&got, "pod-a", Sticky::None, false);
}

#[test]
fn p2c_survives_an_out_of_range_rnd() {
    let f = fleet_of([
        fresh_worker("pod-a"),
        with_in_flight(fresh_worker("pod-b"), 8),
    ]);
    let got = pick(&req_no_run(), &f, PickPolicy::P2c, |n| n + 7);
    assert_placed(&got, "pod-a", Sticky::None, false);
}

#[test]
fn parse_pick_policy_cases() {
    let cases = [
        ("prefix_then_load", Some(PickPolicy::PrefixThenLoad)),
        ("least_loaded", Some(PickPolicy::LeastLoaded)),
        ("p2c", Some(PickPolicy::P2c)),
        (" P2C ", Some(PickPolicy::P2c)),
        ("Least_Loaded", Some(PickPolicy::LeastLoaded)),
        ("", None),
        ("random", None),
    ];
    for (input, want) in cases {
        assert_eq!(
            parse_pick_policy(input),
            want,
            "parse_pick_policy({input:?})"
        );
    }
}

#[test]
fn label_strings() {
    assert_eq!(PickPolicy::PrefixThenLoad.as_str(), "prefix_then_load");
    assert_eq!(PickPolicy::LeastLoaded.as_str(), "least_loaded");
    assert_eq!(PickPolicy::P2c.as_str(), "p2c");
    assert_eq!(Sticky::None.as_str(), "none");
    assert_eq!(Sticky::New.as_str(), "new");
    assert_eq!(Sticky::Hit.as_str(), "hit");
    assert_eq!(Sticky::BrokenLoad.as_str(), "broken_load");
    assert_eq!(Sticky::BrokenShed.as_str(), "broken_shed");
    assert_eq!(Sticky::BrokenGone.as_str(), "broken_gone");
}

const POOL: u64 = 51_000;
const HISTORY: u64 = 14_500;

#[test]
fn sticky_ignores_the_runs_own_history() {
    // The audit case: one run with 14.5k tokens of history on a 51k pool, nothing in flight, the
    // other worker idle. With its own history counted the bound worker reads 0.28 load units
    // over the idle one, past the 0.25 slack, and the run bounces. Without it, the bound worker
    // is as idle as the other and the run stays.
    let bound = |used| with_pool(fresh_worker("pod-a"), POOL, used);
    let cases = [
        (
            "only its own history on an idle fleet stays",
            bound(HISTORY),
            "pod-a",
            Sticky::Hit,
        ),
        (
            // 0.29 of the pool is other runs.
            "unrelated KV over the slack moves",
            bound(HISTORY + 15_000),
            "pod-b",
            Sticky::BrokenLoad,
        ),
        (
            // 5/16 = 0.31
            "in-flight requests over the slack move",
            with_in_flight(bound(HISTORY), 5),
            "pod-b",
            Sticky::BrokenLoad,
        ),
        (
            // 0.125 + 0.098
            "unrelated load within the slack stays",
            with_in_flight(bound(HISTORY + 5_000), 2),
            "pod-a",
            Sticky::Hit,
        ),
    ];
    for (name, bound, want_pod, want_sticky) in cases {
        let mut f = bound_to(
            "pod-a",
            fleet_of([bound, with_pool(fresh_worker("pod-b"), POOL, 0)]),
        );
        f.history = HISTORY;
        let got = pick(&req_run("run1"), &f, PickPolicy::PrefixThenLoad, no_rnd);
        assert_eq!(
            (got.pod(), got.sticky),
            (Some(want_pod), want_sticky),
            "{name}"
        );
        assert!(!got.unknown(), "{name}");
    }
}

#[test]
fn history_only_discounts_the_bound_worker() {
    // least_loaded and p2c ignore history: the bound worker with only its own history is still
    // the busier one, so they move the run.
    for policy in [PickPolicy::LeastLoaded, PickPolicy::P2c] {
        let mut f = bound_to(
            "pod-a",
            fleet_of([
                with_pool(fresh_worker("pod-a"), POOL, HISTORY),
                with_pool(fresh_worker("pod-b"), POOL, 0),
            ]),
        );
        f.history = HISTORY;
        let got = pick(&req_run("run1"), &f, policy, |_| 0);
        assert_placed(&got, "pod-b", Sticky::BrokenLoad, false);
    }
}

#[test]
fn load_without() {
    let w = WorkerView {
        pod: "pod-a".to_owned(),
        kv_pool_tokens: 1000,
        kv_usage: 0.5,
        in_flight: 4,
        max_inflight: 16,
        ..WorkerView::default()
    };
    let no_pool = WorkerView {
        in_flight: 2,
        max_inflight: 16,
        ..WorkerView::default()
    };
    let cases = [
        ("nothing discounted is load", &w, 0, 0.75),
        ("part of the KV", &w, 200, 0.55),
        ("all of the KV", &w, 500, 0.25),
        (
            "more than the KV clamps at the in-flight term",
            &w,
            900,
            0.25,
        ),
        ("a pool of 0 discounts nothing", &no_pool, 100, 1.125),
    ];
    for (name, w, tokens, want) in cases {
        let got = w.load_without(tokens);
        assert!(
            (got - want).abs() <= 1e-9,
            "{name}: load_without({tokens}) = {got}, want {want}"
        );
    }
    assert_eq!(w.load(), 0.75);
}
