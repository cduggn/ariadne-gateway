use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ariadne_decide::{DataClass, Priority, Sticky};
use bytes::Bytes;
use http::{HeaderMap, StatusCode};
use serde_json::Value;
use tokio_util::sync::{CancellationToken, DropGuard};

use super::wire::usage_of;
use super::{Event, Options, Reason, Server};
use crate::clock::Clock;
use crate::fake::{self, Served, overflow as fakeoverflow, vllm};
use crate::fleet::{self, Fleet, Gate, Phase, TenantLimit, WorkerConfig};
use crate::goduration::GoDuration;
use crate::hop::{self, Endpoint, Hopper};
use crate::log::Logger;
use crate::overflow::{self, ApiKey, Backend, Overflow, Superlinked};

/// A step-1 request shaped like the doctor's, and the body the warm-up probes replay.
const DOCTOR_BODY: &str = r#"{"model":"Qwen/Qwen3-8B-AWQ","messages":[{"role":"system","content":"You are cluster-doctor, an SRE agent. Diagnose the cluster from the card and the tool results, then call submit_diagnosis."},{"role":"user","content":"Cluster card: 3 nodes, 2 GPUs, 41 pods, kube-prometheus installed. Task: investigate why checkout-api restarts every few minutes."}],"tools":[{"type":"function","function":{"name":"submit_diagnosis","description":"Submit the final diagnosis","parameters":{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]}}}],"tool_choice":"auto","max_completion_tokens":768,"temperature":0}"#;

/// `DOCTOR_BODY` with the step-1 exchange appended to messages, so the two share a long prefix.
fn doctor_step2() -> String {
    DOCTOR_BODY.replacen(
        r#"],"tools""#,
        r#",{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"kubectl_get","arguments":"{\"kind\":\"pod\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"checkout-api-7d9f OOMKilled, restarts 14"}],"tools""#,
        1,
    )
}

const PODS: [&str; 2] = ["vllm-0", "vllm-1"];

fn other(pod: &str) -> &'static str {
    if pod == PODS[0] { PODS[1] } else { PODS[0] }
}

/// What one test changes from the shipped shape. `hop`, when set, turns on the KV hop over the
/// two fakes.
#[derive(Default)]
struct Tuning {
    config: Option<fn(&mut fleet::Config)>,
    options: Options,
    fakes: Option<fn(&mut vllm::Settings)>,
    hop: Option<hop::Config>,
}

/// Two fake workers, a Gate, a Fleet and a Server, all live until the harness drops.
struct Harness {
    fakes: BTreeMap<&'static str, vllm::Worker>,
    fake: BTreeMap<&'static str, Served>,
    gate: Gate,
    fleet: Fleet,
    url: String,
    events: Arc<Mutex<Vec<Event>>>,
    client: reqwest::Client,
    _server: Served,
    _stop: DropGuard,
}

struct Resp {
    status: u16,
    header: HeaderMap,
    body: Bytes,
}

impl Resp {
    fn header(&self, name: &str) -> &str {
        self.header
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
    }
}

fn doctor_headers(id: &str) -> [(&'static str, String); 7] {
    [
        ("Content-Type", "application/json".to_owned()),
        ("Authorization", "Bearer EMPTY".to_owned()),
        ("X-Request-Id", id.to_owned()),
        ("X-Tenant", "platform".to_owned()),
        ("X-App", "cluster-doctor".to_owned()),
        ("X-Priority", "interactive".to_owned()),
        ("X-Data-Class", "restricted".to_owned()),
    ]
}

async fn start(tn: Tuning) -> Harness {
    let mut fakes = BTreeMap::new();
    let mut fake = BTreeMap::new();
    let mut urls = BTreeMap::new();
    for pod in PODS {
        let w = vllm::Worker::new();
        if let Some(edit) = tn.fakes {
            w.set(edit);
        }
        let served = fake::serve(w.router()).await.expect("serve fake");
        urls.insert(pod.to_owned(), served.url());
        fakes.insert(pod, w);
        fake.insert(pod, served);
    }

    let clock = Clock::system();
    let mut cfg = fleet::Config::defaults(PODS.iter().map(|p| (*p).to_owned()).collect());
    if let Some(edit) = tn.config {
        edit(&mut cfg);
    }
    let gate = Gate::new(cfg, clock.clone(), |_| 0);
    let fleet = Fleet::new(
        gate.clone(),
        urls.clone(),
        WorkerConfig {
            interval: Duration::from_millis(5),
            scrape_timeout: Duration::from_millis(200),
            down_after: Duration::from_millis(200),
            window: Duration::from_secs(1),
            rising_over: Duration::from_millis(20),
            warm_body: Bytes::from_static(DOCTOR_BODY.as_bytes()),
            warm_under: Duration::from_millis(50),
            warm_passes: 2,
            probe_timeout: Duration::from_secs(1),
        },
        reqwest::Client::new(),
        clock.clone(),
    );
    let mut opt = tn.options;
    if let Some(cfg) = tn.hop {
        let endpoints: HashMap<String, Endpoint> = urls
            .iter()
            .map(|(pod, u)| {
                (
                    pod.clone(),
                    Endpoint {
                        base_url: u.clone(),
                        bootstrap_url: u.clone(),
                    },
                )
            })
            .collect();
        let now = clock.clone();
        let hopper = Arc::new(
            Hopper::new(
                cfg,
                endpoints,
                reqwest::Client::new(),
                Arc::new(move || now.now().epoch_offset()),
            )
            .expect("hop config"),
        );
        let forget = Arc::clone(&hopper);
        fleet.on_not_ready(move |pod| forget.forget(pod)); // as main wires it
        opt.hop = Some(hopper);
    }
    let stop = CancellationToken::new();
    tokio::spawn({
        let (fleet, stop) = (fleet.clone(), stop.clone());
        async move { fleet.run(stop).await }
    });

    let events = Arc::new(Mutex::new(Vec::new()));
    opt.log = Logger::discard();
    opt.on_request = Some(Arc::new({
        let events = Arc::clone(&events);
        move |ev: &Event| events.lock().expect("events").push(ev.clone())
    }));
    let server = fake::serve(Server::new(gate.clone(), fleet.clone(), urls, opt).router())
        .await
        .expect("serve gateway");
    Harness {
        fakes,
        fake,
        gate,
        fleet,
        url: server.url(),
        events,
        client: reqwest::Client::new(),
        _server: server,
        _stop: stop.drop_guard(),
    }
}

async fn eventually<F, Fut>(what: &str, mut cond: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline {
        if cond().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    panic!("timed out waiting for {what}");
}

async fn read(resp: reqwest::Response) -> Resp {
    Resp {
        status: resp.status().as_u16(),
        header: resp.headers().clone(),
        body: resp.bytes().await.unwrap_or_default(),
    }
}

fn error_code(body: &[u8]) -> (String, String) {
    let v: Value = serde_json::from_slice(body)
        .unwrap_or_else(|e| panic!("error body is not JSON: {e}\n{body:?}"));
    let s = |k: &str| v["error"][k].as_str().unwrap_or_default().to_owned();
    (s("type"), s("code"))
}

fn pod(ev: &Event) -> &str {
    ev.placement.as_ref().map_or("", |p| p.pod.as_str())
}

fn sticky(ev: &Event) -> Option<Sticky> {
    ev.placement.as_ref().map(|p| p.sticky)
}

impl Harness {
    async fn ready(&self) {
        eventually("/readyz 200", || async {
            self.get("/readyz").await.status == 200
        })
        .await;
    }

    async fn get(&self, path: &str) -> Resp {
        let resp = self
            .client
            .get(format!("{}{path}", self.url))
            .send()
            .await
            .unwrap_or_else(|e| panic!("GET {path}: {e}"));
        read(resp).await
    }

    /// Sends `body` with the doctor's headers under `id`.
    async fn post(&self, id: &str, body: &str) -> Resp {
        self.post_as(id, "restricted", body).await
    }

    /// `post` with a different data class.
    async fn post_as(&self, id: &str, class: &str, body: &str) -> Resp {
        let mut req = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
            .body(body.to_owned());
        for (k, v) in doctor_headers(id) {
            req = req.header(k, if k == "X-Data-Class" { class } else { &v });
        }
        read(
            req.send()
                .await
                .unwrap_or_else(|e| panic!("POST {id}: {e}")),
        )
        .await
    }

    fn recorded_count(&self) -> usize {
        self.fakes.values().map(|w| w.requests().len()).sum()
    }

    /// The newest chat request across both fakes with the given request id.
    fn last_recorded(&self, id: &str) -> vllm::Recorded {
        self.fakes
            .values()
            .find_map(|w| {
                w.requests()
                    .into_iter()
                    .rev()
                    .find(|r| r.header.get("x-request-id").is_some_and(|v| v == id))
            })
            .unwrap_or_else(|| panic!("no fake recorded request {id}"))
    }

    fn events_seen(&self) -> Vec<Event> {
        self.events.lock().expect("events").clone()
    }

    /// The Event for a request id, waiting for the handler to finish it.
    async fn event(&self, id: &str) -> Event {
        let mut found = None;
        eventually(&format!("event for {id}"), || {
            found = self.events_seen().into_iter().find(|e| e.request_id == id);
            let seen = found.is_some();
            async move { seen }
        })
        .await;
        found.expect("found")
    }

    async fn all_ready(&self) {
        eventually("both workers ready", || {
            let ready = self.gate.view().iter().all(|ws| ws.ready);
            async move { ready }
        })
        .await;
    }

    /// Makes `busy` 0.5 load units busier than `idle`, over the 0.25 slack but above the KV
    /// line, and waits for the gate to see it. A run bound to `busy` then moves for load, and a
    /// new run lands on `idle`.
    async fn load(&self, busy: &str, idle: &str) {
        self.fakes[busy].set(|s| s.kv_usage = 0.7);
        self.fakes[idle].set(|s| s.kv_usage = 0.2);
        eventually("the gate to see the new load", || {
            let v: HashMap<String, f64> = self
                .gate
                .view()
                .into_iter()
                .map(|ws| (ws.view.pod.clone(), ws.view.free_ratio()))
                .collect();
            let seen = v[busy] < 0.35 && v[idle] > 0.75;
            async move { seen }
        })
        .await;
    }

    /// Takes `pod` Down and brings it back with a new engine id, as a restarted vLLM has, and
    /// waits for it to be Ready again.
    async fn restart(&self, pod: &str, engine: &str) {
        self.fakes[pod].set(|s| s.metrics_down = true);
        eventually(&format!("{pod} down"), || {
            let down = self
                .fleet
                .status()
                .iter()
                .any(|st| st.pod == pod && st.phase == Phase::Down);
            async move { down }
        })
        .await;
        self.fakes[pod].set(|s| {
            s.metrics_down = false;
            s.engine_id = engine.to_owned();
        });
        self.all_ready().await;
    }

    fn in_flight(&self) -> usize {
        self.gate.view().iter().map(|ws| ws.view.in_flight).sum()
    }

    async fn settled(&self, what: &str) {
        eventually(what, || {
            let idle = self.in_flight() == 0;
            async move { idle }
        })
        .await;
    }

    /// Makes both fake workers report a full KV cache and waits for the gate to see it, so every
    /// new request is a kv_free 503.
    async fn saturate(&self) {
        for w in self.fakes.values() {
            w.set(|s| s.kv_usage = 0.95);
        }
        eventually("both scrapes at 0.95", || {
            let full = self.gate.view().iter().all(|ws| ws.view.kv_usage >= 0.95);
            async move { full }
        })
        .await;
    }
}

/// The remote_engine_id a recorded destination request was told to pull from, "" when it
/// carried no hop.
fn pulled_engine(rec: &vllm::Recorded) -> String {
    let v: Value = serde_json::from_slice(&rec.body).expect("recorded body is JSON");
    v["kv_transfer_params"]["remote_engine_id"]
        .as_str()
        .unwrap_or("")
        .to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_forwards_body_unchanged_and_annotates() {
    let h = start(Tuning::default()).await;
    h.ready().await;

    let resp = h.post("audit-42-s1", DOCTOR_BODY).await;
    assert_eq!(resp.status, 200, "body {:?}", resp.body);
    let rec = h.last_recorded("audit-42-s1");
    assert_eq!(
        rec.body,
        DOCTOR_BODY.as_bytes(),
        "the fake received the client's body"
    );
    assert_eq!(
        Some(&resp.body),
        rec.response.as_ref(),
        "the client received the fake's response"
    );
    for (k, want) in doctor_headers("audit-42-s1") {
        if k == "Content-Type" {
            continue;
        }
        assert_eq!(
            rec.header.get(k).and_then(|v| v.to_str().ok()),
            Some(want.as_str()),
            "fake saw {k}"
        );
    }
    let pod_header = resp.header("X-Pod");
    assert!(PODS.contains(&pod_header), "X-Pod = {pod_header:?}");
    assert_eq!(resp.header("X-Policy"), "prefix_then_load");
    assert_eq!(resp.header("X-Sticky"), "new");
    let timing: Vec<&str> = resp.header("Server-Timing").split(", ").collect();
    assert!(
        timing.len() == 3
            && timing[0].starts_with("gateway;dur=")
            && timing[1].starts_with("queue;dur=")
            && timing[2].starts_with("upstream;dur="),
        "Server-Timing = {timing:?}"
    );
    assert_ne!(resp.header("X-Gateway-Queue-Ms"), "");
    let ev = h.event("audit-42-s1").await;
    let usage = ev.usage.clone().unwrap_or_default();
    assert!(
        ev.status == Some(200)
            && ev.reason.is_none()
            && pod(&ev) == pod_header
            && usage.completion_tokens == 35
            && usage.finish_reason == "tool_calls",
        "event = {ev:?}, want 200 on {pod_header} with the fake's usage"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn second_step_sticks_to_same_pod_with_cached_prefix() {
    let h = start(Tuning::default()).await;
    h.ready().await;

    let first = h.post("audit-7-s1", DOCTOR_BODY).await;
    let second = h.post("audit-7-s2", &doctor_step2()).await;
    assert_eq!((first.status, second.status), (200, 200));
    assert_eq!(first.header("X-Pod"), second.header("X-Pod"));
    assert_eq!(second.header("X-Sticky"), "hit");
    let ev = h.event("audit-7-s2").await;
    assert!(
        sticky(&ev) == Some(Sticky::Hit)
            && ev.usage.as_ref().is_some_and(|u| u.cached_tokens > 0)
            && ev.step == 2,
        "step 2 event = {ev:?}, want sticky hit with cached tokens"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_body_is_400_and_never_reaches_a_worker() {
    let h = start(Tuning::default()).await;
    h.ready().await;
    let before = h.recorded_count();

    let resp = h.post("audit-1-s1", r#"{"model":"m","messages":["#).await;
    let (typ, code) = error_code(&resp.body);
    assert_eq!(
        (resp.status, typ.as_str(), code.as_str()),
        (400, "invalid_request_error", "bad_json")
    );
    assert_eq!(
        h.recorded_count(),
        before,
        "a worker saw the malformed request"
    );
    let ev = h.event("audit-1-s1").await;
    assert!(
        ev.status == Some(400)
            && ev.reason.map(Reason::as_str) == Some("bad_json")
            && pod(&ev).is_empty(),
        "event = {ev:?}, want 400 bad_json with no pod"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_is_400() {
    let h = start(Tuning::default()).await;
    h.ready().await;

    let body = DOCTOR_BODY.replacen(r#""temperature":0"#, r#""temperature":0,"stream":true"#, 1);
    let resp = h.post("audit-2-s1", &body).await;
    let (_, code) = error_code(&resp.body);
    assert_eq!((resp.status, code.as_str()), (400, "stream_unsupported"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tenant_exhausted_is_429() {
    let h = start(Tuning {
        config: Some(|c| {
            c.tenants.insert(
                "platform".to_owned(),
                TenantLimit {
                    rate_per_s: 1.0,
                    burst: 10.0,
                },
            );
        }),
        ..Tuning::default()
    })
    .await;
    h.ready().await;

    let resp = h.post("audit-3-s1", DOCTOR_BODY).await;
    let (typ, code) = error_code(&resp.body);
    assert_eq!(
        (resp.status, typ.as_str(), code.as_str()),
        (429, "rate_limit_exceeded", "tenant_tokens")
    );
    assert_eq!(resp.header("Retry-After"), "60");
    assert_eq!(resp.header("X-Gateway-Reason"), "tenant_tokens");
    let ev = h.event("audit-3-s1").await;
    assert!(
        ev.status == Some(429)
            && ev.reason.map(Reason::as_str) == Some("tenant_tokens")
            && pod(&ev).is_empty()
            && ev.overflow.is_none(),
        "event = {ev:?}, want 429 tenant_tokens with no pod and no overflow decision"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kv_full_on_every_worker_is_503() {
    let h = start(Tuning::default()).await;
    h.ready().await;
    h.saturate().await;

    let resp = h.post("audit-4-s1", DOCTOR_BODY).await;
    let (typ, code) = error_code(&resp.body);
    assert_eq!(
        (resp.status, typ.as_str(), code.as_str()),
        (503, "overloaded", "kv_free")
    );
    assert_eq!(resp.header("Retry-After"), "2");
    assert_eq!(resp.header("X-Gateway-Reason"), "kv_free");
    let ev = h.event("audit-4-s1").await;
    assert_eq!(
        ev.overflow,
        Some(overflow::Outcome::Blocked),
        "restricted 503 overflow"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upstream_500_is_relayed_and_settled() {
    let h = start(Tuning::default()).await;
    h.ready().await;
    for w in h.fakes.values() {
        w.set(|s| s.status = StatusCode::INTERNAL_SERVER_ERROR);
    }

    let resp = h.post("audit-5-s1", DOCTOR_BODY).await;
    assert_eq!(resp.status, 500);
    assert_eq!(
        Some(&resp.body),
        h.last_recorded("audit-5-s1").response.as_ref(),
        "the client received the fake's error body"
    );
    let ev = h.event("audit-5-s1").await;
    assert!(
        ev.status == Some(500) && ev.reason.is_none() && !pod(&ev).is_empty(),
        "event = {ev:?}, want 500 with a pod"
    );
    h.settled("in-flight back to 0").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upstream_timeout_is_502_and_settled() {
    let h = start(Tuning {
        options: Options {
            upstream_timeout: Duration::from_millis(50),
            ..Options::default()
        },
        ..Tuning::default()
    })
    .await;
    h.ready().await;
    for w in h.fakes.values() {
        w.set(|s| s.latency = Duration::from_millis(300));
    }

    let resp = h.post("audit-6-s1", DOCTOR_BODY).await;
    let (typ, code) = error_code(&resp.body);
    assert_eq!(
        (resp.status, typ.as_str(), code.as_str()),
        (502, "upstream_error", "upstream_error")
    );
    let ev = h.event("audit-6-s1").await;
    assert!(
        ev.status == Some(502)
            && ev.reason == Some(Reason::UpstreamError)
            && !pod(&ev).is_empty()
            && ev.upstream >= Duration::from_millis(50),
        "event = {ev:?}, want 502 upstream_error with a pod and the time spent"
    );
    h.settled("in-flight back to 0").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_sees_every_request_once() {
    let h = start(Tuning {
        config: Some(|c| {
            c.tenants.insert(
                "tiny".to_owned(),
                TenantLimit {
                    rate_per_s: 1.0,
                    burst: 10.0,
                },
            );
        }),
        ..Tuning::default()
    })
    .await;
    h.ready().await;

    let ok = h.post("audit-8-s1", DOCTOR_BODY).await;
    h.post("audit-9-s1", "not json").await;
    h.client
        .post(format!("{}/v1/chat/completions", h.url))
        .header("X-Request-Id", "audit-10-s1")
        .header("X-Tenant", "tiny")
        .body(DOCTOR_BODY)
        .send()
        .await
        .expect("POST audit-10-s1");

    h.event("audit-10-s1").await;
    let got = h.events_seen();
    assert_eq!(got.len(), 3, "events: {got:?}");
    let want = [
        ("audit-8-s1", Some(200), None, ok.header("X-Pod")),
        ("audit-9-s1", Some(400), Some("bad_json"), ""),
        ("audit-10-s1", Some(429), Some("tenant_tokens"), ""),
    ];
    for (ev, (id, status, reason, want_pod)) in got.iter().zip(want) {
        assert!(
            ev.request_id == id
                && ev.status == status
                && ev.reason.map(Reason::as_str) == reason
                && pod(ev) == want_pod,
            "event = {ev:?}, want {id} {status:?} {reason:?} {want_pod}"
        );
    }
    assert!(
        got[2].tenant == "tiny" && got[2].priority == Priority::Batch,
        "tenant event = {:?}, want tenant tiny at batch priority",
        got[2]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn readyz_and_debug_endpoints() {
    let h = start(Tuning {
        fakes: Some(|s| s.latency = Duration::from_millis(100)),
        ..Tuning::default()
    })
    .await;
    assert_eq!(h.get("/readyz").await.status, 503, "/readyz before warm-up");
    let healthz = h.get("/healthz").await;
    assert_eq!((healthz.status, &healthz.body[..]), (200, &b"ok"[..]));

    for w in h.fakes.values() {
        w.set(|s| s.latency = Duration::ZERO);
    }
    h.ready().await;

    let direct = h
        .client
        .get(format!("{}/metrics", h.fake["vllm-0"].url()))
        .send()
        .await
        .expect("fake metrics");
    let want = direct.bytes().await.expect("fake metrics body");
    let got = h.get("/debug/workers/vllm-0/metrics").await;
    assert!(
        got.status == 200
            && got.body == want
            && got.header("Content-Type").starts_with("text/plain"),
        "/debug/workers/vllm-0/metrics = {} {}, want the fake's text",
        got.status,
        got.header("Content-Type")
    );
    assert_eq!(
        h.get("/debug/workers/nope/metrics").await.status,
        404,
        "unknown pod metrics"
    );

    // /readyz needs one warm worker, so the second may still be warming here.
    let all_ready = |w: &Value| {
        w.as_array()
            .is_some_and(|w| w.len() == 2 && w.iter().all(|w| w["phase"] == "ready"))
    };
    eventually("both workers ready", || async {
        all_ready(&serde_json::from_slice(&h.get("/debug/workers").await.body).expect("JSON"))
    })
    .await;
    let workers: Value = serde_json::from_slice(&h.get("/debug/workers").await.body)
        .expect("/debug/workers is JSON");
    assert_eq!(workers[0]["pod"], "vllm-0");
    assert_eq!(workers[1]["pod"], "vllm-1");
    for w in workers.as_array().into_iter().flatten() {
        assert!(
            w["phase"] == "ready"
                && w["view"]["Ready"] == true
                && w["view"]["View"]["KVPoolTokens"] == 79056,
            "worker {w}, want ready with the fake's pool"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_route_serves_the_exposition_when_set() {
    let exposition = "# TYPE orch_requests_total counter\norch_requests_total{class=\"restricted\",priority=\"batch\"} 0\n";
    let h = start(Tuning {
        options: Options {
            metrics: Some(Arc::new(move || exposition.to_owned())),
            ..Options::default()
        },
        ..Tuning::default()
    })
    .await;
    let got = h.get("/metrics").await;
    assert_eq!((got.status, &got.body[..]), (200, exposition.as_bytes()));
    assert_eq!(
        got.header("Content-Type"),
        "text/plain; version=0.0.4; charset=utf-8; escaping=underscores"
    );
    assert_eq!(
        start(Tuning::default()).await.get("/metrics").await.status,
        404
    );
}

/// Hops any moved history, so the test exercises the transport rather than the cost rule.
const EAGER_HOP: hop::Config = hop::Config {
    min_tokens: 1,
    shared_prefix_tokens: 0,
    kv_bytes_per_token: 1,
    transfer_bytes_per_s: 1e12,
    prefill_tokens_per_s: 1.0,
    overhead: GoDuration::from_nanos(0),
    timeout: GoDuration::from_nanos(1_000_000_000),
    receive_timeout: GoDuration::from_nanos(1_000_000_000),
    max_inflight: 4,
};

fn hopping() -> Tuning {
    Tuning {
        hop: Some(EAGER_HOP),
        ..Tuning::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn moved_run_hops_its_kv_and_the_new_worker_pulls_it() {
    let h = start(hopping()).await;
    h.ready().await;

    let first = h.post("runA-s1", DOCTOR_BODY).await;
    assert_eq!(first.status, 200, "step 1");
    let from = first.header("X-Pod").to_owned();
    let to = other(&from);
    h.load(&from, to).await;

    let second = h.post("runA-s2", &doctor_step2()).await;
    assert_eq!(
        (
            second.status,
            second.header("X-Pod"),
            second.header("X-Hop")
        ),
        (200, to, "hopped")
    );
    let ev = h.event("runA-s2").await;
    assert!(
        sticky(&ev) == Some(Sticky::BrokenLoad)
            && ev.hop == Some(hop::Outcome::Hopped)
            && ev.hop_time > Duration::ZERO,
        "event = {ev:?}, want broken_load, hopped and a hop time"
    );
    let usage = ev.usage.clone().unwrap_or_default();
    assert!(
        usage.cached_tokens == usage.prompt_tokens && usage.prompt_tokens != 0,
        "cached {} of {} prompt tokens, want the whole pulled history",
        usage.cached_tokens,
        usage.prompt_tokens
    );

    let hold = h.fakes[from.as_str()]
        .requests()
        .pop()
        .expect("the source saw the hold");
    let hold_id = hold
        .header
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        !hold_id.contains("runA"),
        "source request id {hold_id:?} carries the client's"
    );
    let held: Value = serde_json::from_slice(&hold.body).expect("hold body");
    assert!(
        held["kv_transfer_params"]["do_remote_decode"] == true && held["max_tokens"] == 1,
        "source request {held}, want do_remote_decode and a one-token cap"
    );
    let pulled: Value =
        serde_json::from_slice(&h.last_recorded("runA-s2").body).expect("destination body");
    let p = &pulled["kv_transfer_params"];
    assert!(
        p["do_remote_prefill"] == true
            && p["remote_engine_id"] == "fake-engine"
            && p["transfer_id"] == held["kv_transfer_params"]["transfer_id"],
        "destination params {p}, want do_remote_prefill from fake-engine under the source's transfer id"
    );
    h.settled("both tickets settled").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kept_run_does_not_hop() {
    let h = start(hopping()).await;
    h.ready().await;
    h.post("runB-s1", DOCTOR_BODY).await;
    let second = h.post("runB-s2", &doctor_step2()).await;
    assert_eq!(
        (second.header("X-Sticky"), second.header("X-Hop")),
        ("hit", "")
    );
    assert_eq!(
        h.last_recorded("runB-s2").body,
        doctor_step2().as_bytes(),
        "a kept run's body changed on the way to the worker"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_kv_transfer_params_are_400_and_never_reach_a_worker() {
    let h = start(hopping()).await;
    h.ready().await;
    let before = h.recorded_count();

    let spoofed = DOCTOR_BODY.replacen(
        r#""temperature":0}"#,
        r#""temperature":0,"kv_transfer_params":{"do_remote_prefill":true,"remote_bootstrap_addr":"http://attacker.example:8998","remote_engine_id":"x","transfer_id":"xfer-someone-else"}}"#,
        1,
    );
    let resp = h.post("runC-s1", &spoofed).await;
    let (typ, code) = error_code(&resp.body);
    assert_eq!(
        (resp.status, typ.as_str(), code.as_str()),
        (400, "invalid_request_error", "kv_transfer_params")
    );
    assert_eq!(
        h.recorded_count(),
        before,
        "a worker saw a client's kv_transfer_params"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_that_leaves_mid_request_is_client_gone_not_an_upstream_error() {
    let h = start(Tuning::default()).await;
    h.ready().await;
    for w in h.fakes.values() {
        // Slow only after warm-up, which must answer within 50 ms.
        w.set(|s| s.latency = Duration::from_secs(2));
    }

    let mut req = h
        .client
        .post(format!("{}/v1/chat/completions", h.url))
        .timeout(Duration::from_millis(100))
        .body(DOCTOR_BODY);
    for (k, v) in doctor_headers("runD-s1") {
        req = req.header(k, v);
    }
    assert!(
        req.send().await.is_err(),
        "request finished before the client gave up; the fake's latency should outlast the client"
    );

    let ev = h.event("runD-s1").await;
    assert!(
        ev.reason == Some(Reason::ClientGone) && ev.status.is_none(),
        "event reason {:?} status {:?}, want client_gone with no status",
        ev.reason,
        ev.status
    );
    assert!(
        ev.upstream >= Duration::from_millis(50),
        "upstream {:?}, want the time the cancelled call ran",
        ev.upstream
    );
    h.settled("the ticket settled").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hopped_request_that_stalls_is_sent_again_without_the_hop() {
    // The 2026-10-12 case (dx-audit-2 step 16): the hold succeeds, the destination pulls from an
    // engine that is gone and hangs. The receive deadline ends the wait, the request goes again
    // without the hop, and the client gets its answer.
    let mut cfg = EAGER_HOP;
    cfg.receive_timeout = GoDuration::from_nanos(50_000_000);
    let h = start(Tuning {
        hop: Some(cfg),
        ..Tuning::default()
    })
    .await;
    h.all_ready().await;

    let from = h
        .post("runE-s1", DOCTOR_BODY)
        .await
        .header("X-Pod")
        .to_owned();
    let to = other(&from);
    h.fakes[to].set(|s| s.pull_latency = Some(Duration::from_secs(2)));
    h.load(&from, to).await;

    let second = h.post("runE-s2", &doctor_step2()).await;
    assert_eq!(
        (
            second.status,
            second.header("X-Pod"),
            second.header("X-Hop")
        ),
        (200, to, "failed")
    );
    let ev = h.event("runE-s2").await;
    assert!(
        sticky(&ev) == Some(Sticky::BrokenLoad)
            && ev.hop == Some(hop::Outcome::Failed)
            && ev.hop_fallback
            && ev.hop_time > Duration::ZERO,
        "event = {ev:?}, want broken_load, failed, a fallback and a hop time"
    );
    assert!(
        ev.upstream >= Duration::from_millis(50) && ev.status == Some(200) && ev.reason.is_none(),
        "event = {ev:?}, want the stalled attempt's time and a clean 200"
    );

    let ids: Vec<String> = h.fakes[to]
        .requests()
        .iter()
        .filter_map(|r| {
            r.header
                .get("x-request-id")?
                .to_str()
                .ok()
                .map(str::to_owned)
        })
        .filter(|id| id.starts_with("runE-s2"))
        .collect();
    assert_eq!(
        ids,
        ["runE-s2", "runE-s2-recompute"],
        "the hopped attempt then the recompute"
    );
    assert_eq!(pulled_engine(&h.last_recorded("runE-s2")), "fake-engine");
    assert_eq!(
        h.last_recorded("runE-s2-recompute").body,
        doctor_step2().as_bytes(),
        "second attempt body changed"
    );
    h.settled("both tickets settled").await;

    // The source's engine id was forgotten: the next hop off it asks the registry again.
    assert_eq!(
        h.fakes[from.as_str()].queries(),
        1,
        "registry queries before the next hop"
    );
    h.fakes[to].set(|s| s.pull_latency = None);
    h.load(to, &from).await;
    assert_eq!(h.post("runF-s1", DOCTOR_BODY).await.header("X-Pod"), from);
    h.load(&from, to).await;
    assert_eq!(
        h.post("runF-s2", &doctor_step2()).await.header("X-Hop"),
        "hopped"
    );
    assert_eq!(
        h.fakes[from.as_str()].queries(),
        2,
        "registry queries after the fallback"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hopped_request_answer_decides_the_fallback() {
    let cases = [
        (
            "a 5xx on the pull recomputes",
            StatusCode::INTERNAL_SERVER_ERROR,
            200,
            "failed",
        ),
        (
            "a 4xx is relayed as it is",
            StatusCode::BAD_REQUEST,
            400,
            "hopped",
        ),
    ];
    for (name, pull_status, want_status, want_hop) in cases {
        let h = start(hopping()).await;
        h.all_ready().await;
        let from = h
            .post("runG-s1", DOCTOR_BODY)
            .await
            .header("X-Pod")
            .to_owned();
        let to = other(&from);
        h.fakes[to].set(|s| s.pull_status = Some(pull_status));
        h.load(&from, to).await;

        let second = h.post("runG-s2", &doctor_step2()).await;
        assert_eq!(
            (second.status, second.header("X-Hop")),
            (want_status, want_hop),
            "{name}"
        );
        let ev = h.event("runG-s2").await;
        assert_eq!(
            ev.hop_fallback,
            want_hop == "failed",
            "{name}: event fallback"
        );
        let n = h.fakes[to]
            .requests()
            .iter()
            .filter(|r| {
                r.header
                    .get("x-request-id")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|id| id.starts_with("runG-s2"))
            })
            .count();
        assert_eq!(
            n,
            if ev.hop_fallback { 2 } else { 1 },
            "{name}: destination attempts"
        );
        h.settled("both tickets settled").await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restarted_source_is_asked_for_its_engine_id_again() {
    // A hop caches the source's engine id. After the source restarts, the id is gone with the
    // engine, so the next hop off it must ask again, or the destination pulls from a dead id.
    let h = start(hopping()).await;
    h.all_ready().await;

    let from = h
        .post("runH-s1", DOCTOR_BODY)
        .await
        .header("X-Pod")
        .to_owned();
    let to = other(&from);
    h.load(&from, to).await;
    h.post("runH-s2", &doctor_step2()).await;
    assert_eq!(
        pulled_engine(&h.last_recorded("runH-s2")),
        "fake-engine",
        "first hop"
    );

    h.restart(&from, "fake-engine-2").await;
    h.load(to, &from).await;
    assert_eq!(
        h.post("runI-s1", DOCTOR_BODY).await.header("X-Pod"),
        from,
        "runI step 1"
    );
    h.load(&from, to).await;
    let second = h.post("runI-s2", &doctor_step2()).await;
    assert_eq!((second.status, second.header("X-Hop")), (200, "hopped"));
    assert_eq!(
        pulled_engine(&h.last_recorded("runI-s2")),
        "fake-engine-2",
        "hop after the restart"
    );
    assert_eq!(
        h.fakes[from.as_str()].queries(),
        2,
        "registry queries: once per boot"
    );
    h.settled("both tickets settled").await;
}

const OVERFLOW_KEY: &str = "test-key";

/// A fake backend and the circuit over it, probed once so it starts available.
async fn with_overflow(
    edit: impl FnOnce(&mut overflow::Config),
) -> (Arc<Overflow>, fakeoverflow::Backend, Served) {
    let fake = fakeoverflow::Backend::new();
    fake.set(|s| s.key = Some(OVERFLOW_KEY.to_owned()));
    let served = fake::serve(fake.router())
        .await
        .expect("serve fake overflow");
    let mut cfg = overflow::Config {
        base_url: served.url(),
        timeout: Duration::from_secs(1),
        ..overflow::Config::default()
    };
    edit(&mut cfg);
    let max_inflight = cfg.max_inflight;
    let backend = Superlinked::new(
        cfg,
        ApiKey::new(OVERFLOW_KEY.to_owned()),
        reqwest::Client::new(),
    )
    .expect("overflow config");
    let clock = Clock::system();
    let ov = Arc::new(Overflow::new(
        Backend::Superlinked(backend),
        Duration::ZERO,
        max_inflight,
        Arc::new(move || clock.now().epoch_offset()),
        Logger::discard().overflow_log(),
    ));
    ov.probe().await;
    (ov, fake, served)
}

fn overflowing(ov: &Arc<Overflow>) -> Tuning {
    Tuning {
        options: Options {
            overflow: Some(Arc::clone(ov)),
            ..Options::default()
        },
        ..Tuning::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saturated_503_of_a_non_restricted_request_is_served_by_the_overflow_backend() {
    let (ov, fake, _backend) =
        with_overflow(|c| c.model = Some("Qwen/Qwen3.8-27B-FP8".to_owned())).await;
    let h = start(overflowing(&ov)).await;
    h.ready().await;
    h.saturate().await;
    let before = h.recorded_count();

    let resp = h.post_as("ovf-1-s1", "internal", DOCTOR_BODY).await;
    assert_eq!(
        resp.status, 200,
        "body {:?}, want 200 from the overflow backend",
        resp.body
    );
    let reqs = fake.requests();
    assert_eq!(reqs.len(), 1, "backend requests");
    assert_eq!(
        Some(&resp.body),
        reqs[0].response.as_ref(),
        "the client received the backend's answer"
    );
    let sent: Value = serde_json::from_slice(&reqs[0].body).expect("sent body");
    assert_eq!(
        sent["model"], "Qwen/Qwen3.8-27B-FP8",
        "the configured model"
    );
    assert_eq!(
        reqs[0]
            .header
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some(format!("Bearer {OVERFLOW_KEY}").as_str()),
        "the configured key, not the client's"
    );
    for k in ["X-Request-Id", "X-Tenant", "X-Data-Class"] {
        assert!(reqs[0].header.get(k).is_none(), "{k} left the box");
    }
    assert_eq!(
        h.recorded_count(),
        before,
        "a worker saw the overflowed request"
    );
    assert_eq!(resp.header("X-Overflow"), "sent");
    assert_eq!(resp.header("X-Overflow-Backend"), "superlinked");
    assert_eq!(
        resp.header("X-Gateway-Reason"),
        "kv_free",
        "the shed reason the request left for"
    );
    assert!(
        resp.header("Server-Timing").starts_with("overflow;dur="),
        "Server-Timing"
    );
    assert_eq!((resp.header("Retry-After"), resp.header("X-Pod")), ("", ""));
    let ev = h.event("ovf-1-s1").await;
    let usage = ev.usage.clone().unwrap_or_default();
    assert!(
        ev.status == Some(200)
            && ev.reason.is_none()
            && pod(&ev).is_empty()
            && ev.overflow == Some(overflow::Outcome::Sent)
            && ev.overflow_time > Duration::ZERO
            && usage.completion_tokens == 30
            && usage.finish_reason == "tool_calls"
            && ev.class == DataClass::Internal,
        "event = {ev:?}, want 200 sent off-box with the backend's usage and no pod"
    );
    assert_eq!(
        h.in_flight(),
        0,
        "nothing was reserved for an overflowed request"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restricted_503_never_reaches_the_overflow_backend() {
    let (ov, fake, _backend) = with_overflow(|_| {}).await;
    let h = start(overflowing(&ov)).await;
    h.ready().await;
    h.saturate().await;

    for (i, class) in ["restricted", "", "RESTRICTED", "nonsense"]
        .into_iter()
        .enumerate()
    {
        let id = format!("ovf-r-s{}", i + 1);
        let resp = h.post_as(&id, class, DOCTOR_BODY).await;
        let (typ, code) = error_code(&resp.body);
        assert_eq!(
            (resp.status, typ.as_str(), code.as_str()),
            (503, "overloaded", "kv_free"),
            "class {class:?}"
        );
        assert_eq!(
            (resp.header("X-Overflow"), resp.header("Retry-After")),
            ("", "2"),
            "class {class:?}: the plain 503"
        );
        let ev = h.event(&id).await;
        assert!(
            ev.overflow == Some(overflow::Outcome::Blocked)
                && ev.class == DataClass::Restricted
                && ev.overflow_time == Duration::ZERO,
            "class {class:?}: event = {ev:?}, want blocked_invariant on a restricted request"
        );
    }
    assert_eq!(
        fake.requests().len(),
        0,
        "the backend saw restricted requests"
    );
    assert!(ov.available(), "blocked requests opened the circuit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unavailable_backend_yields_the_original_503_unchanged() {
    // The control: no backend at all, the shipped gateway.
    let control = start(Tuning::default()).await;
    control.ready().await;
    control.saturate().await;
    let want = control.post_as("ovf-c-s1", "internal", DOCTOR_BODY).await;
    assert_eq!(want.status, 503, "control status");
    assert_eq!(
        control.event("ovf-c-s1").await.overflow,
        Some(overflow::Outcome::NoBackend),
        "control overflow"
    );

    let (ov, fake, _backend) = with_overflow(|_| {}).await;
    fake.set(|s| s.down = true);
    ov.probe().await; // the backend went away: the circuit opens
    let h = start(overflowing(&ov)).await;
    h.ready().await;
    h.saturate().await;

    let got = h.post_as("ovf-u-s1", "internal", DOCTOR_BODY).await;
    assert_eq!(
        (got.status, &got.body),
        (want.status, &want.body),
        "the control's answer"
    );
    for k in ["Retry-After", "X-Gateway-Reason", "Content-Type"] {
        assert_eq!(got.header(k), want.header(k), "{k}");
    }
    for k in ["X-Overflow", "X-Overflow-Backend", "Server-Timing"] {
        assert_eq!(got.header(k), "", "{k} on the 503");
    }
    assert_eq!(
        fake.requests().len(),
        0,
        "an unavailable backend saw requests"
    );
    let ev = h.event("ovf-u-s1").await;
    assert!(
        ev.status == Some(503)
            && ev.reason.map(Reason::as_str) == Some("kv_free")
            && ev.overflow == Some(overflow::Outcome::Unavailable),
        "event = {ev:?}, want 503 kv_free marked unavailable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_error_yields_the_original_503_and_opens_the_circuit() {
    let (ov, fake, _backend) = with_overflow(|_| {}).await;
    fake.set(|s| s.status = StatusCode::PAYMENT_REQUIRED);
    let h = start(overflowing(&ov)).await;
    h.ready().await;
    h.saturate().await;

    let first = h.post_as("ovf-e-s1", "internal", DOCTOR_BODY).await;
    let (_, code) = error_code(&first.body);
    assert_eq!(
        (first.status, code.as_str(), first.header("X-Overflow")),
        (503, "kv_free", ""),
        "the plain 503"
    );
    let ev = h.event("ovf-e-s1").await;
    assert!(
        ev.overflow == Some(overflow::Outcome::Failed)
            && ev.overflow_time > Duration::ZERO
            && ev.status == Some(503),
        "event = {ev:?}, want 503 marked failed with the time spent"
    );
    let second = h.post_as("ovf-e-s2", "internal", DOCTOR_BODY).await;
    assert_eq!(second.status, 503);
    assert_eq!(
        h.event("ovf-e-s2").await.overflow,
        Some(overflow::Outcome::Unavailable),
        "the circuit should be open"
    );
    assert_eq!(fake.requests().len(), 1, "backend requests");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_429_never_reaches_the_overflow_backend() {
    let (ov, fake, _backend) = with_overflow(|_| {}).await;
    let mut tn = overflowing(&ov);
    tn.config = Some(|c| {
        c.tenants.insert(
            "platform".to_owned(),
            TenantLimit {
                rate_per_s: 1.0,
                burst: 10.0,
            },
        );
    });
    let h = start(tn).await;
    h.ready().await;
    let resp = h.post_as("ovf-q-s1", "internal", DOCTOR_BODY).await;
    assert_eq!(
        (resp.status, resp.header("X-Overflow")),
        (429, ""),
        "a plain 429"
    );
    assert_eq!(
        h.event("ovf-q-s1").await.overflow,
        None,
        "no overflow decision for a 429"
    );
    assert_eq!(
        fake.requests().len(),
        0,
        "backend requests for a tenant refusal"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admitted_request_never_reaches_the_overflow_backend() {
    let (ov, fake, _backend) = with_overflow(|_| {}).await;
    let h = start(overflowing(&ov)).await;
    h.ready().await;
    let resp = h.post_as("ovf-a-s1", "internal", DOCTOR_BODY).await;
    assert!(
        resp.status == 200
            && !resp.header("X-Pod").is_empty()
            && resp.header("X-Overflow").is_empty(),
        "got {} pod {:?} X-Overflow {:?}, want 200 from a worker",
        resp.status,
        resp.header("X-Pod"),
        resp.header("X-Overflow")
    );
    assert_eq!(
        fake.requests().len(),
        0,
        "backend requests while a worker had room"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overflowed_answers_are_charged_to_the_tenant() {
    const BURST: f64 = 50000.0;
    let (ov, fake, _backend) = with_overflow(|_| {}).await;
    let mut tn = overflowing(&ov);
    tn.config = Some(|c| {
        c.tenants.insert(
            "platform".to_owned(),
            TenantLimit {
                rate_per_s: 0.0,
                burst: BURST,
            },
        );
    });
    let h = start(tn).await;
    h.ready().await;
    h.saturate().await;
    assert_eq!(h.gate.tenant_tokens("platform"), BURST, "bucket before");

    let resp = h.post_as("chg-1-s1", "internal", DOCTOR_BODY).await;
    assert_eq!((resp.status, resp.header("X-Overflow")), (200, "sent"));
    let u = usage_of(&resp.body).unwrap_or_default();
    let want = BURST - (u.prompt_tokens + u.completion_tokens) as f64;
    assert_eq!(
        h.gate.tenant_tokens("platform"),
        want,
        "the off-box usage must be charged like a local one"
    );

    fake.set(|s| s.status = StatusCode::UNPROCESSABLE_ENTITY);
    let resp = h.post_as("chg-2-s1", "internal", DOCTOR_BODY).await;
    assert_eq!(
        resp.status, 503,
        "the original 503 when the backend refuses the request"
    );
    assert_eq!(
        h.gate.tenant_tokens("platform"),
        want,
        "a refused request is never charged"
    );
    assert!(
        ov.available(),
        "one request's 422 opened the circuit for every tenant"
    );

    h.gate.set_tenant_tokens_for_test("platform", 100.0);
    let resp = h.post_as("chg-3-s1", "internal", DOCTOR_BODY).await;
    let reason = h.event("chg-3-s1").await.reason;
    assert!(
        resp.status == 429 && reason.map(Reason::as_str) == Some("tenant_tokens"),
        "status {}, reason {reason:?}, want 429 tenant_tokens: a tenant that spent its quota off-box is refused at the gate like any other",
        resp.status
    );
    assert_eq!(
        fake.requests().len(),
        2,
        "a request over quota never leaves"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routing_errors_match_go_servemux() {
    let h = start(Tuning::default()).await;
    let not_found = h.get("/nope").await;
    assert_eq!(
        (
            not_found.status,
            &not_found.body[..],
            not_found.header("X-Content-Type-Options")
        ),
        (404, &b"404 page not found\n"[..], "nosniff")
    );
    let wrong = h.get("/v1/chat/completions").await;
    assert_eq!(
        (wrong.status, &wrong.body[..], wrong.header("Allow")),
        (405, &b"Method Not Allowed\n"[..], "POST")
    );
    let unknown = h.get("/debug/workers/nope/metrics").await;
    assert_eq!(
        (unknown.status, &unknown.body[..]),
        (404, &b"unknown pod nope\n"[..])
    );
}
