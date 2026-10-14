use std::sync::Mutex;
use std::time::Instant;

use ariadne_decide::{DataClass, Priority, Reason, Refusal, Request, Timestamp, may_leave};

use super::superlinked::rewrite_model;
use super::*;
use crate::fake::overflow::Backend as Fake;
use crate::fake::{Served, serve};

const BODY: &str = r#"{"model":"Qwen/Qwen3-8B-AWQ","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":8}"#;

fn request(class: DataClass) -> Request {
    Request {
        id: "run-s1".to_owned(),
        run: None,
        step: 0,
        tenant: "platform".to_owned(),
        app: String::new(),
        priority: Priority::Interactive,
        class,
        body_bytes: BODY.len(),
        max_out: 8,
        arrived: Timestamp::from_epoch_offset(Duration::ZERO),
        deadline: Timestamp::from_epoch_offset(Duration::from_secs(10)),
    }
}

fn permit(class: DataClass) -> Offboxable {
    let refusal = Refusal {
        reason: Reason::KvFree,
        retry_after: Duration::from_secs(1),
        detail: String::new(),
    };
    may_leave(&request(class), &refusal).expect("a non-restricted 503 may leave")
}

fn body() -> Bytes {
    Bytes::from_static(BODY.as_bytes())
}

fn clock() -> Clock {
    let start = Instant::now();
    Arc::new(move || start.elapsed())
}

type Event = (Level, String, Vec<(String, Value)>);

fn capture() -> (Log, Arc<Mutex<Vec<Event>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let log: Log = Arc::new(move |level, msg, attrs| {
        let attrs = attrs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        sink.lock().unwrap().push((level, msg.to_owned(), attrs));
    });
    (log, events)
}

fn superlinked(cfg: Config, key: &str) -> Superlinked {
    Superlinked::new(cfg, ApiKey::new(key.to_owned()), reqwest::Client::new()).unwrap()
}

struct Fixture {
    ov: Arc<Overflow>,
    fake: Fake,
    events: Arc<Mutex<Vec<Event>>>,
    _served: Served,
}

/// A fake backend behind a Superlinked client and an Overflow with no
/// periodic probing, at the given key.
async fn fixture(mut cfg: Config, key: &str) -> Fixture {
    let fake = Fake::new();
    let served = serve(fake.router()).await.unwrap();
    cfg.base_url = served.url();
    let max_inflight = cfg.max_inflight;
    let (log, events) = capture();
    let backend = Backend::Superlinked(superlinked(cfg, key));
    Fixture {
        ov: Arc::new(Overflow::new(
            backend,
            Duration::ZERO,
            max_inflight,
            clock(),
            log,
        )),
        fake,
        events,
        _served: served,
    }
}

fn unreachable_config() -> Config {
    Config {
        base_url: "http://127.0.0.1:1".to_owned(),
        timeout: Duration::from_secs(1),
        max_inflight: 1,
        ..Config::default()
    }
}

#[test]
fn config_validate() {
    assert_eq!(Config::default().validate(), Ok(()));
    let base = |url: &str| Config {
        base_url: url.to_owned(),
        ..Config::default()
    };
    let bad: [(&str, Config, &str); 10] = [
        (
            "no scheme",
            base("api.superlinked.com"),
            r#"base url "api.superlinked.com" is not an http(s) origin"#,
        ),
        (
            "ftp",
            base("ftp://x"),
            r#"base url "ftp://x" is not an http(s) origin"#,
        ),
        (
            "no host",
            base("http://"),
            r#"base url "http://" is not an http(s) origin"#,
        ),
        (
            "empty authority",
            base("http:///x"),
            r#"base url "http:///x" is not an http(s) origin"#,
        ),
        (
            "space",
            base("http://a b"),
            r#"base url "http://a b" is not an http(s) origin"#,
        ),
        (
            "newline",
            base("http://x\n"),
            r#"base url "http://x\n" is not an http(s) origin"#,
        ),
        (
            "relative chat path",
            Config {
                chat_path: "v1/chat/completions".to_owned(),
                ..Config::default()
            },
            "chat path and probe path must start with /",
        ),
        (
            "relative probe path",
            Config {
                probe_path: "v1/models".to_owned(),
                ..Config::default()
            },
            "chat path and probe path must start with /",
        ),
        (
            "zero timeout",
            Config {
                timeout: Duration::ZERO,
                ..Config::default()
            },
            "timeout must be positive",
        ),
        (
            "no slots",
            Config {
                max_inflight: 0,
                ..Config::default()
            },
            "max inflight must be at least 1",
        ),
    ];
    for (name, cfg, want) in bad {
        let err = cfg.validate().expect_err(name);
        assert_eq!(err.to_string(), want, "{name}");
    }
    assert_eq!(base("HTTP://X").validate(), Ok(()));
    let err = Superlinked::new(base("ftp://x"), None, reqwest::Client::new()).unwrap_err();
    assert_eq!(
        err.to_string(),
        r#"overflow config: base url "ftp://x" is not an http(s) origin"#
    );
}

#[tokio::test]
async fn origin_drops_a_trailing_slash_or_v1() {
    for base_url in [
        "https://api.superlinked.com",
        "https://api.superlinked.com/",
        "https://api.superlinked.com/v1",
        "https://api.superlinked.com/v1/",
    ] {
        let fake = Fake::new();
        let served = serve(fake.router()).await.unwrap();
        let cfg = Config {
            base_url: base_url.replace("https://api.superlinked.com", &served.url()),
            ..Config::default()
        };
        assert!(superlinked(cfg, "").probe().await.is_ok(), "{base_url}");
        assert_eq!(fake.probes(), 1, "{base_url}");
    }
}

#[tokio::test]
async fn backend_starts_unavailable_until_a_probe_passes() {
    let f = fixture(Config::default(), "k").await;
    assert!(!f.ov.available(), "available before any probe");
    let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(attempt.outcome(), Outcome::Unavailable);
    assert!(
        f.fake.requests().is_empty(),
        "the backend saw a request while unavailable"
    );

    f.fake.set(|s| s.key = Some("k".to_owned()));
    assert!(f.ov.probe().await && f.ov.available());
    assert_eq!((f.fake.probes(), f.ov.probes()), (1, 1));
    assert_eq!(
        *f.events.lock().unwrap(),
        [(
            Level::Info,
            "overflow backend available".to_owned(),
            vec![("backend".to_owned(), Value::from("superlinked"))]
        )]
    );
}

#[tokio::test]
async fn probe_fails_on_wrong_key_and_on_down_backend() {
    let f = fixture(Config::default(), "wrong").await;
    f.fake.set(|s| s.key = Some("k".to_owned()));
    assert!(!f.ov.probe().await, "probe passed with the wrong key");
    assert_eq!(
        *f.events.lock().unwrap(),
        [(
            Level::Debug,
            "overflow backend still unavailable".to_owned(),
            vec![
                ("backend".to_owned(), Value::from("superlinked")),
                (
                    "err".to_owned(),
                    Value::from("probe /v1/models answered 401")
                ),
            ]
        )]
    );

    let f2 = fixture(Config::default(), "").await;
    f2.fake.set(|s| s.down = true);
    assert!(!f2.ov.probe().await, "probe passed against a down backend");

    let backend = Backend::Superlinked(superlinked(unreachable_config(), ""));
    let (log, _) = capture();
    let ov3 = Overflow::new(backend, Duration::ZERO, 1, clock(), log);
    assert!(!ov3.probe().await, "probe passed against nothing");
}

#[tokio::test]
async fn a_failed_probe_after_a_pass_warns() {
    let f = fixture(Config::default(), "").await;
    assert!(f.ov.probe().await);
    f.fake.set(|s| s.down = true);
    assert!(!f.ov.probe().await);
    let events = f.events.lock().unwrap();
    assert_eq!(
        events[1],
        (
            Level::Warn,
            "overflow backend unavailable".to_owned(),
            vec![
                ("backend".to_owned(), Value::from("superlinked")),
                (
                    "err".to_owned(),
                    Value::from("probe /v1/models answered 503")
                ),
            ]
        )
    );
}

#[tokio::test]
async fn sent_relays_the_backend_answer_with_the_model_rewritten() {
    let model = "Qwen/Qwen3.8-27B-FP8";
    let cfg = Config {
        model: Some(model.to_owned()),
        ..Config::default()
    };
    let f = fixture(cfg, "k").await;
    f.fake.set(|s| s.key = Some("k".to_owned()));
    f.ov.probe().await;

    let attempt = f.ov.send(permit(DataClass::Public), body()).await;
    let Attempt::Sent {
        content_type,
        body,
        took,
    } = attempt
    else {
        panic!("Send = {attempt:?}, want sent");
    };
    assert!(took > Duration::ZERO);
    let reqs = f.fake.requests();
    assert_eq!(reqs.len(), 1);
    assert_eq!(Some(&body), reqs[0].response.as_ref());
    assert!(
        content_type
            .unwrap()
            .as_bytes()
            .starts_with(b"application/json")
    );
    assert_eq!(
        reqs[0].body,
        r#"{"max_completion_tokens":8,"messages":[{"role":"user","content":"hi"}],"model":"Qwen/Qwen3.8-27B-FP8"}"#
    );
    assert_eq!(reqs[0].header["authorization"], "Bearer k");
    for name in [
        "x-request-id",
        "x-tenant",
        "x-data-class",
        "x-priority",
        "x-app",
    ] {
        assert!(!reqs[0].header.contains_key(name), "{name} left the box");
    }
}

#[tokio::test]
async fn without_a_model_the_clients_model_is_forwarded_unchanged() {
    let f = fixture(Config::default(), "").await;
    f.ov.probe().await;
    let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(attempt.outcome(), Outcome::Sent, "{attempt:?}");
    assert_eq!(f.fake.requests()[0].body, BODY);
    assert!(!f.fake.requests()[0].header.contains_key("authorization"));
}

#[test]
fn rewrite_model_writes_what_go_json_marshal_writes() {
    let body = "{ \"z\": [1, 2.50, {\"b\": \"a<b>&c \u{2028}\", \"a\": null}],\n \"model\" : \"old\", \"messages\": \"x\\u003cy\", \"a&\": true, \"model\": \"dup\" }";
    let out = rewrite_model(body.as_bytes(), "new<m>").unwrap();
    assert_eq!(
        out,
        r#"{"aU+0026":true,"messages":"xU+003cy","model":"newU+003cmU+003e","z":[1,2.50,{"b":"aU+003cbU+003eU+0026c U+2028","a":null}]}"#.replace("U+", "\\u")
    );
    let err = rewrite_model(b"[1]", "m").unwrap_err();
    assert!(
        err.to_string()
            .starts_with("overflow: body is not a JSON object: "),
        "{err}"
    );
}

#[tokio::test]
async fn a_restricted_request_gets_no_permit_and_never_reaches_the_backend() {
    let f = fixture(Config::default(), "").await;
    f.ov.probe().await;
    let refusal = Refusal {
        reason: Reason::KvFree,
        retry_after: Duration::from_secs(1),
        detail: String::new(),
    };
    assert!(may_leave(&request(DataClass::Restricted), &refusal).is_none());
    assert!(f.fake.requests().is_empty());
    assert!(f.ov.available());
}

#[tokio::test]
async fn backend_error_fails_and_opens_the_circuit_until_the_next_probe() {
    let f = fixture(Config::default(), "").await;
    f.ov.probe().await;
    f.fake.set(|s| s.status = StatusCode::PAYMENT_REQUIRED);

    let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
    let Attempt::Failed { error, .. } = &attempt else {
        panic!("Send against a 402 backend = {attempt:?}, want failed");
    };
    assert_eq!(error.to_string(), "overflow: backend answered 402");
    assert!(!f.ov.available(), "a failed send left the circuit closed");
    let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(attempt.outcome(), Outcome::Unavailable);
    assert_eq!(f.fake.requests().len(), 1, "the open circuit must not send");

    f.fake.set(|s| s.status = StatusCode::OK);
    f.ov.probe().await;
    let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(attempt.outcome(), Outcome::Sent);
}

#[tokio::test]
async fn request_shaped_4xx_fails_without_opening_the_circuit() {
    let f = fixture(Config::default(), "").await;
    f.ov.probe().await;
    for status in [
        StatusCode::BAD_REQUEST,
        StatusCode::PAYLOAD_TOO_LARGE,
        StatusCode::UNPROCESSABLE_ENTITY,
    ] {
        f.fake.set(|s| s.status = status);
        let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
        assert_eq!(attempt.outcome(), Outcome::Failed, "{status}");
        assert!(
            f.ov.available(),
            "a {status} about one request opened the circuit"
        );
    }
    f.fake.set(|s| s.status = StatusCode::OK);
    let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(attempt.outcome(), Outcome::Sent);
}

#[tokio::test]
async fn which_failures_open_the_circuit() {
    let dial = reqwest::Client::new()
        .get("http://127.0.0.1:1")
        .send()
        .await
        .unwrap_err();
    let status = |code: u16| SendError::Status(StatusCode::from_u16(code).unwrap());
    let cases = [
        (SendError::Transport(dial), Fault::Backend),
        (
            SendError::Body(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            Fault::Request,
        ),
        (SendError::Timeout { url: String::new() }, Fault::Backend),
        (status(401), Fault::Backend),
        (status(402), Fault::Backend),
        (status(403), Fault::Backend),
        (status(429), Fault::Backend),
        (status(500), Fault::Backend),
        (status(503), Fault::Backend),
        (status(400), Fault::Request),
        (status(404), Fault::Request),
        (status(413), Fault::Request),
        (status(422), Fault::Request),
    ];
    for (error, want) in cases {
        assert_eq!(error.fault(), want, "{error}");
    }
}

#[tokio::test]
async fn timeout_fails_within_the_configured_bound() {
    let cfg = Config {
        timeout: Duration::from_millis(50),
        ..Config::default()
    };
    let f = fixture(cfg, "").await;
    f.ov.probe().await;
    f.fake.set(|s| s.latency = Duration::from_secs(1));
    let attempt = f.ov.send(permit(DataClass::Internal), body()).await;
    let took = attempt.took();
    assert_eq!(attempt.outcome(), Outcome::Failed, "{attempt:?}");
    assert!(
        took >= Duration::from_millis(50) && took <= Duration::from_millis(500),
        "took {took:?}, want about 50 ms"
    );
    let Attempt::Failed { error, .. } = attempt else {
        unreachable!()
    };
    assert_eq!(
        error.to_string(),
        format!(
            "Post \"{}/v1/chat/completions\": context deadline exceeded",
            f._served.url()
        )
    );
}

#[tokio::test]
async fn client_gone_does_not_open_the_circuit() {
    let f = fixture(Config::default(), "").await;
    f.ov.probe().await;
    f.fake.set(|s| s.latency = Duration::from_secs(1));
    let send = f.ov.send(permit(DataClass::Internal), body());
    assert!(
        tokio::time::timeout(Duration::from_millis(20), send)
            .await
            .is_err()
    );
    assert!(f.ov.available(), "a client that left opened the circuit");
}

async fn until_the_backend_has_a_request(fake: &Fake) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while fake.requests().is_empty() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

#[tokio::test]
async fn inflight_cap() {
    let cfg = Config {
        max_inflight: 1,
        ..Config::default()
    };
    let f = fixture(cfg, "").await;
    f.ov.probe().await;
    f.fake.set(|s| s.latency = Duration::from_millis(200));
    let ov = Arc::clone(&f.ov);
    let first = tokio::spawn(async move { ov.send(permit(DataClass::Internal), body()).await });
    until_the_backend_has_a_request(&f.fake).await;
    let second = f.ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(second.outcome(), Outcome::Capped);
    assert_eq!(first.await.unwrap().outcome(), Outcome::Sent);
    assert!(f.ov.available(), "a capped send opened the circuit");
}

#[tokio::test]
async fn a_dropped_send_frees_its_slot() {
    let cfg = Config {
        max_inflight: 1,
        ..Config::default()
    };
    let f = fixture(cfg, "").await;
    f.ov.probe().await;
    f.fake.set(|s| s.latency = Duration::from_millis(200));
    let ov = Arc::clone(&f.ov);
    let first = tokio::spawn(async move { ov.send(permit(DataClass::Internal), body()).await });
    until_the_backend_has_a_request(&f.fake).await;
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    let next = f.ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(next.outcome(), Outcome::Sent);
}

#[tokio::test]
async fn run_probes_at_start_and_then_periodically() {
    let fake = Fake::new();
    let served = serve(fake.router()).await.unwrap();
    let cfg = Config {
        base_url: served.url(),
        ..Config::default()
    };
    let (log, _) = capture();
    let backend = Backend::Superlinked(superlinked(cfg, ""));
    let ov = Arc::new(Overflow::new(
        backend,
        Duration::from_millis(10),
        1,
        clock(),
        log,
    ));
    let shutdown = CancellationToken::new();
    let run = tokio::spawn({
        let ov = Arc::clone(&ov);
        let shutdown = shutdown.clone();
        async move { ov.run(shutdown).await }
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while ov.probes() < 3 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    shutdown.cancel();
    run.await.unwrap();
    assert!(
        ov.probes() >= 3 && ov.available(),
        "probes = {}",
        ov.probes()
    );
}

#[tokio::test(start_paused = true)]
async fn run_without_an_interval_probes_once_and_waits_for_shutdown() {
    let (log, _) = capture();
    let ov = Arc::new(Overflow::new(
        Backend::Null,
        Duration::ZERO,
        1,
        clock(),
        log,
    ));
    let shutdown = CancellationToken::new();
    let run = tokio::spawn({
        let ov = Arc::clone(&ov);
        let shutdown = shutdown.clone();
        async move { ov.run(shutdown).await }
    });
    tokio::time::sleep(Duration::from_secs(3600)).await;
    assert_eq!(ov.probes(), 1);
    assert!(!run.is_finished());
    shutdown.cancel();
    run.await.unwrap();
}

#[tokio::test]
async fn null_backend() {
    let (log, _) = capture();
    let ov = Overflow::new(Backend::Null, Duration::ZERO, 1, clock(), log);
    assert!(!ov.probe().await && !ov.available());
    assert_eq!(ov.name(), "null");
    let attempt = ov.send(permit(DataClass::Internal), body()).await;
    assert_eq!(attempt.outcome(), Outcome::Unavailable);
    let err = Backend::Null
        .send(permit(DataClass::Internal), body())
        .await
        .unwrap_err();
    assert!(matches!(err, SendError::NoBackend), "{err:?}");
}

#[test]
fn outcomes_are_gos_labels() {
    let labels = Outcome::ALL.map(Outcome::as_str);
    assert_eq!(
        labels,
        [
            "blocked_invariant",
            "no_backend",
            "sent",
            "failed",
            "unavailable",
            "capped",
            "quota"
        ]
    );
}

#[tokio::test]
async fn the_key_never_prints() {
    const KEY: &str = "sk-secret-7f3a";
    let f = fixture(Config::default(), KEY).await;
    f.fake.set(|s| s.key = Some("other".to_owned()));
    f.ov.probe().await;
    f.fake.set(|s| s.key = Some(KEY.to_owned()));
    f.ov.probe().await;
    f.fake.set(|s| s.key = Some("other".to_owned()));
    let refused = f.ov.send(permit(DataClass::Internal), body()).await;

    let unreachable = superlinked(unreachable_config(), KEY);
    let dial = unreachable
        .send(permit(DataClass::Internal), body())
        .await
        .unwrap_err();
    let probe = unreachable.probe().await.unwrap_err();

    let printed = [
        format!("{:?}", f.ov),
        format!("{unreachable:?}"),
        format!("{:?}", ApiKey::new(KEY.to_owned())),
        format!("{refused:?}"),
        format!("{dial} {dial:?}"),
        format!("{probe} {probe:?}"),
        format!("{:?}", f.events.lock().unwrap()),
    ];
    assert!(printed[0].contains("ApiKey(<redacted>)"), "{}", printed[0]);
    for text in printed {
        assert!(!text.contains(KEY), "the key leaked: {text}");
    }
}
