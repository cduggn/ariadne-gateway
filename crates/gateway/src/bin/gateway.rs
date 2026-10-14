//! The inference gateway. It guards, admits, places and queues chat completions in front of a
//! fixed set of vLLM workers and forwards each body unchanged. Every flag reads its default from
//! the environment variable GW_<NAME>, upper-cased with dashes turned into underscores.
//!
//! Flags are parsed by hand to keep Go's `flag` package behaviour and messages: `-flag` and
//! `--flag`, `=value` or a separate value, parsing stopping at the first non-flag argument, and
//! Go's error text and usage listing.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use ariadne_decide::{PickPolicy, parse_pick_policy};
use ariadne_gateway::clock::Clock;
use ariadne_gateway::fleet::{self, Fleet, Gate, WorkerConfig};
use ariadne_gateway::goduration::GoDuration;
use ariadne_gateway::hop::{self, Endpoint, Hopper};
use ariadne_gateway::log::{self, Logger, Threshold, go_atoi};
use ariadne_gateway::metrics::Metrics;
use ariadne_gateway::overflow::{self, ApiKey, Backend, Overflow, Superlinked};
use ariadne_gateway::serve::{self, Server};
use bytes::Bytes;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

const DEFAULT_WORKERS: &str = "vllm-0=http://vllm-0.vllm.default.svc.cluster.local:8000,\
                               vllm-1=http://vllm-1.vllm.default.svc.cluster.local:8000";

/// How long a stopping gateway waits for requests in flight.
const DRAIN: Duration = Duration::from_secs(30);

/// A startup failure: printed as `gateway: <msg>`, exit status 2.
struct Fatal(String);

impl<T: Into<String>> From<T> for Fatal {
    fn from(msg: T) -> Self {
        Self(msg.into())
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let program = argv
        .first()
        .cloned()
        .unwrap_or_else(|| "gateway".to_owned());
    let mut flags = match Flags::define() {
        Ok(flags) => flags,
        Err(Fatal(msg)) => return fatal(&msg),
    };
    match flags.parse(argv.get(1..).unwrap_or_default()) {
        Ok(()) => {}
        Err(ParseFail::Help) => {
            eprint!("{}", flags.usage(&program));
            return ExitCode::SUCCESS;
        }
        Err(ParseFail::Bad(msg)) => {
            eprint!("{msg}\n{}", flags.usage(&program));
            return ExitCode::from(2);
        }
    }
    let settings = match Settings::from_flags(flags) {
        Ok(s) => s,
        Err(Fatal(msg)) => return fatal(&msg),
    };
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => return fatal(&format!("runtime: {e}")),
    };
    match runtime.block_on(run(settings)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fatal(msg)) => fatal(&msg),
    }
}

fn fatal(msg: &str) -> ExitCode {
    eprintln!("gateway: {msg}");
    ExitCode::from(2)
}

/// Everything the flags said, checked.
struct Settings {
    listen: String,
    urls: BTreeMap<String, String>,
    policy: PickPolicy,
    max_inflight: i64,
    max_queued: i64,
    warm: Bytes,
    pool: String,
    shared_prefix: i64,
    log: Logger,
    hop: Option<(hop::Config, i64)>,
    overflow: Option<OverflowSettings>,
}

struct OverflowSettings {
    cfg: overflow::Config,
    /// As given, for the startup line; the config holds them clamped to what can run.
    timeout: GoDuration,
    probe_interval: GoDuration,
    max_inflight: i64,
    key_env: String,
    key: Option<ApiKey>,
}

impl Settings {
    /// Go's checks after `flag.Parse`, in Go's order.
    fn from_flags(f: Flags) -> Result<Self, Fatal> {
        let urls = parse_workers(&f.workers.value)?;
        let policy = parse_pick_policy(&f.policy.value)
            .ok_or_else(|| format!("unknown policy {:?}", f.policy.value))?;
        if f.warm_body.value.is_empty() {
            return Err("--warm-body is required".into());
        }
        let warm = read_file(&f.warm_body.value).map_err(|e| format!("read warm body: {e}"))?;
        let threshold = Threshold::parse(&f.log_level.value)
            .ok_or_else(|| format!("bad log level {:?}", f.log_level.value))?;
        let hop = f.hop.value.then(|| {
            let cfg = hop::Config {
                min_tokens: f.hop_min_tokens.value,
                shared_prefix_tokens: f.shared_prefix_tokens.value,
                kv_bytes_per_token: f.hop_kv_bytes_per_token.value,
                transfer_bytes_per_s: f.hop_transfer_bytes_per_s.value,
                prefill_tokens_per_s: f.hop_prefill_tokens_per_s.value,
                overhead: f.hop_overhead.value,
                timeout: f.hop_timeout.value,
                receive_timeout: f.hop_receive_timeout.value,
                max_inflight: f.hop_max_inflight.value,
            };
            (cfg, f.hop_bootstrap_port.value)
        });
        let overflow = f.overflow.value.then(|| {
            let key = std::env::var(&f.overflow_api_key_env.value).unwrap_or_default();
            OverflowSettings {
                cfg: overflow::Config {
                    base_url: f.overflow_base_url.value.clone(),
                    chat_path: f.overflow_chat_path.value.clone(),
                    probe_path: f.overflow_probe_path.value.clone(),
                    model: Some(f.overflow_model.value.clone()).filter(|m| !m.is_empty()),
                    timeout: f.overflow_timeout.value.to_std(),
                    probe_interval: f.overflow_probe_interval.value.to_std(),
                    max_inflight: usize::try_from(f.overflow_max_inflight.value).unwrap_or(0),
                },
                timeout: f.overflow_timeout.value,
                probe_interval: f.overflow_probe_interval.value,
                max_inflight: f.overflow_max_inflight.value,
                key_env: f.overflow_api_key_env.value.clone(),
                key: ApiKey::new(key),
            }
        });
        Ok(Self {
            listen: f.listen.value,
            urls,
            policy,
            max_inflight: f.max_inflight.value,
            max_queued: f.max_queued.value,
            warm: Bytes::from(warm),
            pool: f.pool.value,
            shared_prefix: f.shared_prefix_tokens.value,
            log: Logger::new(threshold, std::io::stdout()),
            hop,
            overflow,
        })
    }
}

async fn run(s: Settings) -> Result<(), Fatal> {
    let log = s.log.clone();
    let clock = Clock::system();
    let pods: Vec<String> = s.urls.keys().cloned().collect();
    let mut cfg = fleet::Config::defaults(pods.clone());
    cfg.policy = s.policy;
    cfg.max_inflight = usize::try_from(s.max_inflight).unwrap_or(0);
    cfg.max_queued = usize::try_from(s.max_queued).unwrap_or(0);
    let gate = Gate::new(cfg, clock.clone(), |n| rand::random_range(0..n.max(1)));
    let workers = Fleet::new(
        gate.clone(),
        s.urls.clone(),
        WorkerConfig::defaults(s.warm),
        reqwest::Client::new(),
        clock.clone(),
    );
    let metrics = Metrics::new(
        gate.clone(),
        workers.clone(),
        &s.pool,
        u64::try_from(s.shared_prefix).unwrap_or(0),
    );
    let (observe, render) = (metrics.clone(), metrics.clone());
    let mut opts = serve::Options {
        log: log.clone(),
        clock: clock.clone(),
        on_request: Some(Arc::new(move |ev| observe.observe(ev))),
        metrics: Some(Arc::new(move || render.render())),
        ..serve::Options::default()
    };

    if let Some((cfg, port)) = s.hop {
        let hopper = Arc::new(new_hopper(cfg, &s.urls, port, &clock)?);
        opts.hop = Some(Arc::clone(&hopper));
        let forget = Arc::clone(&hopper);
        // A worker that left Ready or restarted has a new engine id.
        workers.on_not_ready(move |pod| forget.forget(pod));
        log.info(
            "kv hop on",
            &[
                ("min_tokens", Value::from(cfg.min_tokens)),
                ("kv_bytes_per_token", Value::from(cfg.kv_bytes_per_token)),
                (
                    "transfer_bytes_per_s",
                    Value::from(cfg.transfer_bytes_per_s),
                ),
                (
                    "prefill_tokens_per_s",
                    Value::from(cfg.prefill_tokens_per_s),
                ),
                ("overhead", Value::from(cfg.overhead.to_string())),
                ("timeout", Value::from(cfg.timeout.to_string())),
                (
                    "receive_timeout",
                    Value::from(cfg.receive_timeout.to_string()),
                ),
                ("max_inflight", Value::from(cfg.max_inflight)),
                ("bootstrap_port", Value::from(port)),
            ],
        );
    }

    let mut ov = None;
    if let Some(o) = s.overflow {
        let key_set = o.key.is_some();
        let base_url = o.cfg.base_url.clone();
        let chat_path = o.cfg.chat_path.clone();
        let probe_path = o.cfg.probe_path.clone();
        let model = o.cfg.model.clone().unwrap_or_default();
        let backend =
            Superlinked::new(o.cfg, o.key, reqwest::Client::new()).map_err(|e| e.to_string())?;
        let now = clock.clone();
        let circuit = Arc::new(Overflow::new(
            Backend::Superlinked(backend),
            o.probe_interval.to_std(),
            usize::try_from(o.max_inflight).unwrap_or(0),
            Arc::new(move || now.now().epoch_offset()),
            log.overflow_log(),
        ));
        opts.overflow = Some(Arc::clone(&circuit));
        metrics.watch_overflow(Arc::clone(&circuit));
        if !key_set {
            log.warn(
                "overflow key is empty; the backend stays unavailable until a probe passes",
                &[("key_env", Value::from(o.key_env.as_str()))],
            );
        }
        log.info(
            "overflow on",
            &[
                ("backend", Value::from(circuit.name())),
                ("base_url", Value::from(base_url)),
                ("chat_path", Value::from(chat_path)),
                ("probe_path", Value::from(probe_path)),
                ("model", Value::from(model)),
                ("timeout", Value::from(o.timeout.to_string())),
                ("probe_interval", Value::from(o.probe_interval.to_string())),
                ("max_inflight", Value::from(o.max_inflight)),
                ("key_env", Value::from(o.key_env.as_str())),
                ("key_set", Value::from(key_set)),
            ],
        );
        ov = Some(circuit);
    }
    let router = Server::new(gate, workers.clone(), s.urls, opts).router();

    let stop = CancellationToken::new();
    let fleet_done = tokio::spawn({
        let stop = stop.clone();
        async move { workers.run(stop).await }
    });
    let overflow_done = tokio::spawn({
        let stop = stop.clone();
        async move {
            match ov {
                // Probes at start and every interval; a failed probe never stops the gateway.
                Some(ov) => ov.run(stop).await,
                None => stop.cancelled().await,
            }
        }
    });

    log.info(
        "gateway listening",
        &[
            ("addr", Value::from(s.listen.as_str())),
            ("pods", Value::from(pods)),
            ("policy", Value::from(s.policy.as_str())),
            ("max_inflight", Value::from(s.max_inflight)),
            ("max_queued", Value::from(s.max_queued)),
            ("pool", Value::from(s.pool.as_str())),
            ("shared_prefix_tokens", Value::from(s.shared_prefix)),
        ],
    );
    let listener = listen(&s.listen)
        .await
        .map_err(|e| format!("serve: listen tcp {}: {e}", s.listen))?;
    let mut served = tokio::spawn(serve_http(listener, router, stop.clone()));
    tokio::select! {
        r = &mut served => {
            if let Ok(Err(e)) = r {
                return Err(format!("serve: {e}").into());
            }
        }
        () = signalled() => {
            stop.cancel();
            if tokio::time::timeout(DRAIN, served).await.is_err() {
                log.warn("shutdown", &[("err", Value::from("context deadline exceeded"))]);
            }
        }
    }
    stop.cancel();
    let _ = fleet_done.await;
    let _ = overflow_done.await;
    log.info("gateway stopped", &[]);
    Ok(())
}

/// Go's `http.Server{ReadHeaderTimeout: 10 * time.Second}`: a client that never finishes its
/// request headers is cut off instead of holding a connection forever.
const READ_HEADER_TIMEOUT: Duration = Duration::from_secs(10);

/// Serves `router` over HTTP/1.1 until `stop`, then waits for open connections to finish. It
/// replaces `axum::serve`, which has no header read timeout.
async fn serve_http(
    listener: TcpListener,
    router: axum::Router,
    stop: CancellationToken,
) -> std::io::Result<()> {
    use hyper_util::rt::{TokioIo, TokioTimer};
    use hyper_util::server::graceful::GracefulShutdown;
    use hyper_util::service::TowerToHyperService;

    let mut http = hyper::server::conn::http1::Builder::new();
    http.timer(TokioTimer::new())
        .header_read_timeout(READ_HEADER_TIMEOUT);
    let graceful = GracefulShutdown::new();
    loop {
        let stream = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                // Out of file descriptors and the like: back off, as axum::serve does.
                Err(_) => {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            },
            () = stop.cancelled() => break,
        };
        let conn = http.serve_connection(
            TokioIo::new(stream),
            TowerToHyperService::new(router.clone()),
        );
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            let _ = conn.await;
        });
    }
    graceful.shutdown().await;
    Ok(())
}

/// SIGINT or SIGTERM.
async fn signalled() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

/// Binds as Go's `net.Listen("tcp", addr)` does: an empty host is every interface, both families
/// where the host has IPv6.
async fn listen(addr: &str) -> Result<TcpListener, String> {
    let describe = |e: std::io::Error| match e.kind() {
        std::io::ErrorKind::AddrInUse => "bind: address already in use".to_owned(),
        _ => e.to_string(),
    };
    match addr.strip_prefix(':') {
        Some(port) => match TcpListener::bind(format!("[::]:{port}")).await {
            Ok(l) => Ok(l),
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Err(describe(e)),
            Err(_) => TcpListener::bind(format!("0.0.0.0:{port}"))
                .await
                .map_err(describe),
        },
        None => TcpListener::bind(addr).await.map_err(describe),
    }
}

/// The KV hop over the workers, each reached for Mooncake on its own host at the bootstrap port.
fn new_hopper(
    cfg: hop::Config,
    urls: &BTreeMap<String, String>,
    port: i64,
    clock: &Clock,
) -> Result<Hopper, Fatal> {
    let mut endpoints = HashMap::new();
    for (pod, base) in urls {
        let port = u16::try_from(port)
            .map_err(|_| format!("hop endpoint for {pod}: bootstrap port {port} out of range"))?;
        let bootstrap =
            hop::bootstrap_url(base, port).map_err(|e| format!("hop endpoint for {pod}: {e}"))?;
        let base_url = base.strip_suffix('/').unwrap_or(base).to_owned();
        endpoints.insert(
            pod.clone(),
            Endpoint {
                base_url,
                bootstrap_url: bootstrap,
            },
        );
    }
    let now = clock.clone();
    Hopper::new(
        cfg,
        endpoints,
        reqwest::Client::new(),
        Arc::new(move || now.now().epoch_offset()),
    )
    .map_err(|e| Fatal(e.to_string()))
}

/// Reads "pod=url,pod=url". A missing "=", an empty side or a repeated pod is an error.
fn parse_workers(s: &str) -> Result<BTreeMap<String, String>, Fatal> {
    let mut urls = BTreeMap::new();
    for item in s.split(',').map(str::trim).filter(|i| !i.is_empty()) {
        let (pod, url) = item
            .split_once('=')
            .filter(|(pod, url)| !pod.is_empty() && !url.is_empty())
            .ok_or_else(|| format!("bad worker {item:?}, want pod=url"))?;
        if urls.insert(pod.to_owned(), url.to_owned()).is_some() {
            return Err(format!("worker {pod:?} listed twice").into());
        }
    }
    if urls.is_empty() {
        return Err("no workers configured".into());
    }
    Ok(urls)
}

/// `os.ReadFile` with Go's error text: the failing call, the path, then the lower-case reason.
fn read_file(path: &str) -> Result<Vec<u8>, String> {
    let go = |op: &str, e: std::io::Error| {
        let text = match e.kind() {
            std::io::ErrorKind::NotFound => "no such file or directory".to_owned(),
            std::io::ErrorKind::PermissionDenied => "permission denied".to_owned(),
            std::io::ErrorKind::IsADirectory => "is a directory".to_owned(),
            _ => {
                let t = e.to_string();
                let t = t.split(" (os error").next().unwrap_or(&t).to_owned();
                let mut c = t.chars();
                c.next()
                    .map(|f| f.to_lowercase().chain(c).collect())
                    .unwrap_or_default()
            }
        };
        format!("{op} {path}: {text}")
    };
    let mut f = std::fs::File::open(path).map_err(|e| go("open", e))?;
    let mut out = Vec::new();
    f.read_to_end(&mut out).map_err(|e| go("read", e))?;
    Ok(out)
}

/// Every flag, in Go's definition order, which is also the order their environment defaults are
/// read and so the order a bad one is reported in.
struct Flags {
    listen: Flag<String>,
    workers: Flag<String>,
    policy: Flag<String>,
    max_inflight: Flag<i64>,
    max_queued: Flag<i64>,
    warm_body: Flag<String>,
    pool: Flag<String>,
    shared_prefix_tokens: Flag<i64>,
    log_level: Flag<String>,
    hop: Flag<bool>,
    hop_min_tokens: Flag<i64>,
    hop_kv_bytes_per_token: Flag<i64>,
    hop_transfer_bytes_per_s: Flag<f64>,
    hop_prefill_tokens_per_s: Flag<f64>,
    hop_overhead: Flag<GoDuration>,
    hop_timeout: Flag<GoDuration>,
    hop_receive_timeout: Flag<GoDuration>,
    hop_max_inflight: Flag<i64>,
    hop_bootstrap_port: Flag<i64>,
    overflow: Flag<bool>,
    overflow_base_url: Flag<String>,
    overflow_chat_path: Flag<String>,
    overflow_probe_path: Flag<String>,
    overflow_model: Flag<String>,
    overflow_api_key_env: Flag<String>,
    overflow_timeout: Flag<GoDuration>,
    overflow_probe_interval: Flag<GoDuration>,
    overflow_max_inflight: Flag<i64>,
}

enum ParseFail {
    Help,
    Bad(String),
}

impl Flags {
    fn define() -> Result<Self, Fatal> {
        let h = hop::DEFAULT_CONFIG;
        let o = overflow::Config::default();
        Ok(Self {
            listen: Flag::new("listen", ":8000".to_owned(), "address to serve on")?,
            workers: Flag::new(
                "workers",
                DEFAULT_WORKERS.to_owned(),
                "comma-separated pod=url list of vLLM workers",
            )?,
            policy: Flag::new(
                "policy",
                PickPolicy::PrefixThenLoad.as_str().to_owned(),
                "pick policy: prefix_then_load, least_loaded or p2c",
            )?,
            max_inflight: Flag::new("max-inflight", 16, "in-flight requests per worker")?,
            max_queued: Flag::new("max-queued", 32, "queued requests per worker")?,
            warm_body: Flag::new(
                "warm-body",
                String::new(),
                "path to the recorded step-1 request JSON that warm-up probes replay (required)",
            )?,
            pool: Flag::new(
                "pool",
                "sliced".to_owned(),
                "pool label on the replica metrics",
            )?,
            shared_prefix_tokens: Flag::new(
                "shared-prefix-tokens",
                3899,
                "tokens every prompt shares, which split cached tokens into shared_hit and run_hit",
            )?,
            log_level: Flag::new(
                "log-level",
                "info".to_owned(),
                "log level: debug, info, warn or error",
            )?,
            hop: Flag::new(
                "hop",
                false,
                "copy a moved run's KV to its new worker through vLLM's MooncakeConnector (workers must run it)",
            )?,
            hop_min_tokens: Flag::new(
                "hop-min-tokens",
                h.min_tokens,
                "shortest run history worth a hop, in tokens",
            )?,
            hop_kv_bytes_per_token: Flag::new(
                "hop-kv-bytes-per-token",
                h.kv_bytes_per_token,
                "the served model's KV bytes per token",
            )?,
            hop_transfer_bytes_per_s: Flag::new(
                "hop-transfer-bytes-per-s",
                h.transfer_bytes_per_s,
                "measured KV copy bandwidth between two workers",
            )?,
            hop_prefill_tokens_per_s: Flag::new(
                "hop-prefill-tokens-per-s",
                h.prefill_tokens_per_s,
                "measured uncached prefill rate of one worker",
            )?,
            hop_overhead: Flag::new("hop-overhead", h.overhead, "fixed cost of one hop")?,
            hop_timeout: Flag::new(
                "hop-timeout",
                h.timeout,
                "longest wait for the old worker before recomputing instead",
            )?,
            hop_receive_timeout: Flag::new(
                "hop-receive-timeout",
                h.receive_timeout,
                "longest wait for the new worker to answer a hopped request before sending it again without the hop",
            )?,
            hop_max_inflight: Flag::new(
                "hop-max-inflight",
                h.max_inflight,
                "hops running at once; past this a moved run recomputes",
            )?,
            hop_bootstrap_port: Flag::new(
                "hop-bootstrap-port",
                i64::from(hop::DEFAULT_BOOTSTRAP_PORT),
                "workers' Mooncake bootstrap port (VLLM_MOONCAKE_BOOTSTRAP_PORT)",
            )?,
            overflow: Flag::new(
                "overflow",
                false,
                "send a 503 that may leave the box to the overflow backend instead of refusing it (restricted requests never leave)",
            )?,
            overflow_base_url: Flag::new(
                "overflow-base-url",
                o.base_url,
                "overflow backend origin, without a path",
            )?,
            overflow_chat_path: Flag::new(
                "overflow-chat-path",
                o.chat_path,
                "chat-completions path on the overflow backend",
            )?,
            overflow_probe_path: Flag::new(
                "overflow-probe-path",
                o.probe_path,
                "path the health probe fetches on the overflow backend; any 2xx passes",
            )?,
            overflow_model: Flag::new(
                "overflow-model",
                String::new(),
                "model name sent to the overflow backend; empty forwards the client's",
            )?,
            overflow_api_key_env: Flag::new(
                "overflow-api-key-env",
                "GW_OVERFLOW_API_KEY".to_owned(),
                "environment variable holding the overflow backend's bearer key",
            )?,
            overflow_timeout: Flag::new(
                "overflow-timeout",
                GoDuration::from(o.timeout),
                "longest wait for the overflow backend before answering the 503",
            )?,
            overflow_probe_interval: Flag::new(
                "overflow-probe-interval",
                GoDuration::from(o.probe_interval),
                "how often the overflow backend is re-probed",
            )?,
            overflow_max_inflight: Flag::new(
                "overflow-max-inflight",
                i64::try_from(o.max_inflight).unwrap_or(i64::MAX),
                "overflow calls running at once; past this a 503 stays a 503",
            )?,
        })
    }

    fn all(&mut self) -> [&mut dyn AnyFlag; 28] {
        [
            &mut self.listen,
            &mut self.workers,
            &mut self.policy,
            &mut self.max_inflight,
            &mut self.max_queued,
            &mut self.warm_body,
            &mut self.pool,
            &mut self.shared_prefix_tokens,
            &mut self.log_level,
            &mut self.hop,
            &mut self.hop_min_tokens,
            &mut self.hop_kv_bytes_per_token,
            &mut self.hop_transfer_bytes_per_s,
            &mut self.hop_prefill_tokens_per_s,
            &mut self.hop_overhead,
            &mut self.hop_timeout,
            &mut self.hop_receive_timeout,
            &mut self.hop_max_inflight,
            &mut self.hop_bootstrap_port,
            &mut self.overflow,
            &mut self.overflow_base_url,
            &mut self.overflow_chat_path,
            &mut self.overflow_probe_path,
            &mut self.overflow_model,
            &mut self.overflow_api_key_env,
            &mut self.overflow_timeout,
            &mut self.overflow_probe_interval,
            &mut self.overflow_max_inflight,
        ]
    }

    /// Go's `FlagSet.Parse`: stops at the first non-flag argument or after "--".
    fn parse(&mut self, args: &[String]) -> Result<(), ParseFail> {
        let mut rest = args.iter();
        while let Some(arg) = rest.as_slice().first() {
            let Some(body) = arg.strip_prefix('-').filter(|b| !b.is_empty()) else {
                return Ok(());
            };
            rest.next();
            let name = match body.strip_prefix('-') {
                Some("") => return Ok(()),
                Some(n) => n,
                None => body,
            };
            if name.is_empty() || name.starts_with(['-', '=']) {
                return Err(ParseFail::Bad(format!("bad flag syntax: {arg}")));
            }
            let (name, inline) = match name.split_once('=') {
                Some((n, v)) => (n, Some(v)),
                None => (name, None),
            };
            let mut all = self.all();
            let Some(flag) = all.iter_mut().find(|f| f.name() == name) else {
                if name == "help" || name == "h" {
                    return Err(ParseFail::Help);
                }
                return Err(ParseFail::Bad(format!(
                    "flag provided but not defined: -{name}"
                )));
            };
            if flag.is_bool() {
                let value = inline.unwrap_or("true");
                flag.set(value).map_err(|e| {
                    ParseFail::Bad(format!("invalid boolean value {value:?} for -{name}: {e}"))
                })?;
                continue;
            }
            let value = match inline {
                Some(v) => v,
                None => rest
                    .next()
                    .ok_or_else(|| ParseFail::Bad(format!("flag needs an argument: -{name}")))?,
            };
            flag.set(value).map_err(|e| {
                ParseFail::Bad(format!("invalid value {value:?} for flag -{name}: {e}"))
            })?;
        }
        Ok(())
    }

    /// Go's `Usage`: the program name, then every flag sorted by name with its default.
    fn usage(&mut self, program: &str) -> String {
        let mut lines: Vec<(&str, String)> = self
            .all()
            .into_iter()
            .map(|f| (f.name(), f.usage_entry()))
            .collect();
        lines.sort_by(|a, b| a.0.cmp(b.0));
        let mut out = format!("Usage of {program}:\n");
        for (_, entry) in lines {
            out.push_str(&entry);
        }
        out
    }
}

struct Flag<T> {
    name: &'static str,
    usage: &'static str,
    /// The default after the environment, which Go's usage listing shows.
    default: T,
    value: T,
}

impl<T: FlagValue> Flag<T> {
    /// Takes the default from GW_<NAME> when set. A set but unparsable value fails at startup
    /// rather than silently taking the built-in default.
    fn new(name: &'static str, builtin: T, usage: &'static str) -> Result<Self, Fatal> {
        let var = format!("GW_{}", name.to_uppercase().replace('-', "_"));
        let default = match std::env::var(&var) {
            Ok(v) => {
                T::from_env(&v).ok_or_else(|| format!("{var}={v:?} is not {}", T::ENV_KIND))?
            }
            Err(_) => builtin,
        };
        Ok(Self {
            name,
            usage,
            value: default.clone(),
            default,
        })
    }
}

/// One flag, whatever its type, as the parser and the usage listing see it.
trait AnyFlag {
    fn name(&self) -> &'static str;
    fn is_bool(&self) -> bool;
    fn set(&mut self, s: &str) -> Result<(), NumError>;
    fn usage_entry(&self) -> String;
}

impl<T: FlagValue> AnyFlag for Flag<T> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn is_bool(&self) -> bool {
        T::TYPE.is_empty()
    }

    fn set(&mut self, s: &str) -> Result<(), NumError> {
        self.value = T::from_flag(s)?;
        Ok(())
    }

    /// `PrintDefaults` for this flag.
    fn usage_entry(&self) -> String {
        let mut out = format!("  -{}", self.name);
        if !T::TYPE.is_empty() {
            out.push(' ');
            out.push_str(T::TYPE);
        }
        out.push_str(if out.len() <= 4 { "\t" } else { "\n    \t" });
        out.push_str(&self.usage.replace('\n', "\n    \t"));
        if let Some(shown) = self.default.shown() {
            out.push_str(&format!(" (default {shown})"));
        }
        out.push('\n');
        out
    }
}

/// strconv's two failure kinds, as the flag package words them.
#[derive(Debug)]
enum NumError {
    Syntax,
    Range,
}

impl std::fmt::Display for NumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Syntax => "parse error",
            Self::Range => "value out of range",
        })
    }
}

/// A flag's type, with Go's parsing of a command-line value and of its environment default.
trait FlagValue: Clone {
    /// The type name in the usage listing; empty for a bool.
    const TYPE: &'static str;
    /// Completes "GW_X=... is not ...".
    const ENV_KIND: &'static str;
    fn from_flag(s: &str) -> Result<Self, NumError>;
    fn from_env(s: &str) -> Option<Self>;
    /// The default as the usage listing prints it, `None` for the type's zero value.
    fn shown(&self) -> Option<String>;
}

impl FlagValue for String {
    const TYPE: &'static str = "string";
    const ENV_KIND: &'static str = "a string";
    fn from_flag(s: &str) -> Result<Self, NumError> {
        Ok(s.to_owned())
    }
    fn from_env(s: &str) -> Option<Self> {
        Some(s.to_owned())
    }
    fn shown(&self) -> Option<String> {
        (!self.is_empty()).then(|| format!("{self:?}"))
    }
}

impl FlagValue for i64 {
    const TYPE: &'static str = "int";
    const ENV_KIND: &'static str = "an integer";
    fn from_flag(s: &str) -> Result<Self, NumError> {
        parse_int_base0(s)
    }
    fn from_env(s: &str) -> Option<Self> {
        go_atoi(s)
    }
    fn shown(&self) -> Option<String> {
        (*self != 0).then(|| self.to_string())
    }
}

impl FlagValue for bool {
    const TYPE: &'static str = "";
    const ENV_KIND: &'static str = "a boolean";
    fn from_flag(s: &str) -> Result<Self, NumError> {
        parse_bool(s).ok_or(NumError::Syntax)
    }
    fn from_env(s: &str) -> Option<Self> {
        parse_bool(s)
    }
    fn shown(&self) -> Option<String> {
        self.then(|| "true".to_owned())
    }
}

impl FlagValue for f64 {
    const TYPE: &'static str = "float";
    const ENV_KIND: &'static str = "a number";
    fn from_flag(s: &str) -> Result<Self, NumError> {
        parse_float(s)
    }
    fn from_env(s: &str) -> Option<Self> {
        parse_float(s).ok()
    }
    fn shown(&self) -> Option<String> {
        (*self != 0.0).then(|| format_g(*self))
    }
}

impl FlagValue for GoDuration {
    const TYPE: &'static str = "duration";
    const ENV_KIND: &'static str = "a duration";
    fn from_flag(s: &str) -> Result<Self, NumError> {
        s.parse().map_err(|_| NumError::Syntax)
    }
    fn from_env(s: &str) -> Option<Self> {
        s.parse().ok()
    }
    fn shown(&self) -> Option<String> {
        (self.nanos() != 0).then(|| self.to_string())
    }
}

/// `strconv.ParseBool`.
fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "1" | "t" | "T" | "TRUE" | "true" | "True" => Some(true),
        "0" | "f" | "F" | "FALSE" | "false" | "False" => Some(false),
        _ => None,
    }
}

/// `strconv.ParseInt(s, 0, 64)`: a sign, an optional 0x, 0o, 0b or leading-0 octal prefix, and
/// underscores between digits.
fn parse_int_base0(s: &str) -> Result<i64, NumError> {
    let (neg, body) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let lower = body.to_ascii_lowercase();
    let (radix, digits, prefixed) = if let Some(d) = lower.strip_prefix("0x") {
        (16, d.to_owned(), true)
    } else if let Some(d) = lower.strip_prefix("0o") {
        (8, d.to_owned(), true)
    } else if let Some(d) = lower.strip_prefix("0b") {
        (2, d.to_owned(), true)
    } else if lower.len() > 1 && lower.starts_with('0') {
        (8, lower[1..].to_owned(), true)
    } else {
        (10, lower, false)
    };
    if !underscores_ok(&digits, prefixed) {
        return Err(NumError::Syntax);
    }
    let digits = digits.replace('_', "");
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return Err(NumError::Syntax);
    }
    let magnitude = u64::from_str_radix(&digits, radix).map_err(|_| NumError::Range)?;
    let limit = if neg { 1u64 << 63 } else { (1u64 << 63) - 1 };
    if magnitude > limit {
        return Err(NumError::Range);
    }
    Ok(if neg {
        0i64.wrapping_sub_unsigned(magnitude)
    } else {
        magnitude as i64
    })
}

/// Go's rule for underscores in a number: each separates two digits, or a base prefix from a
/// digit.
fn underscores_ok(digits: &str, prefixed: bool) -> bool {
    let b = digits.as_bytes();
    b.iter().enumerate().all(|(i, &c)| {
        c != b'_' || {
            let before = if i == 0 { prefixed } else { b[i - 1] != b'_' };
            before && b.get(i + 1).is_some_and(|&n| n != b'_')
        }
    })
}

/// `strconv.ParseFloat(s, 64)` for decimal input. Go's hexadecimal floats are not accepted.
fn parse_float(s: &str) -> Result<f64, NumError> {
    let unsigned = s.strip_prefix(['+', '-']).unwrap_or(s).to_ascii_lowercase();
    let special = matches!(unsigned.as_str(), "inf" | "infinity" | "nan");
    if !special && !underscores_ok(&unsigned, false) {
        return Err(NumError::Syntax);
    }
    let f: f64 = s.replace('_', "").parse().map_err(|_| NumError::Syntax)?;
    if f.is_infinite() && !special {
        return Err(NumError::Range);
    }
    Ok(f)
}

/// `strconv.FormatFloat(f, 'g', -1, 64)`, which the flag package shows a float default in.
fn format_g(f: f64) -> String {
    let e = format!("{f:e}");
    let (mantissa, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..6).contains(&exp) {
        log::go_float(f)
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_parse_as_go_flags_do() {
        let cases: [(&str, Option<i64>); 10] = [
            ("16", Some(16)),
            ("-3", Some(-3)),
            ("0x10", Some(16)),
            ("010", Some(8)),
            ("0b101", Some(5)),
            ("1_000", Some(1000)),
            ("1__0", None),
            ("_1", None),
            ("many", None),
            ("9223372036854775808", None),
        ];
        for (s, want) in cases {
            assert_eq!(parse_int_base0(s).ok(), want, "{s}");
        }
    }

    #[test]
    fn floats_format_as_go_g() {
        assert_eq!(format_g(2e9), "2e+09");
        assert_eq!(format_g(3800.0), "3800");
        assert_eq!(format_g(1e12), "1e+12");
        assert_eq!(format_g(0.0001), "0.0001");
        assert_eq!(format_g(0.00001), "1e-05");
        assert_eq!(format_g(123456.0), "123456");
        assert_eq!(format_g(1234567.0), "1.234567e+06");
    }

    #[test]
    fn workers_parse_with_go_errors() {
        assert_eq!(
            parse_workers(" a=http://x , b=http://y,")
                .ok()
                .map(|m| m.len()),
            Some(2)
        );
        let err = |s| parse_workers(s).err().map(|Fatal(m)| m);
        assert_eq!(
            err("vllm-0").as_deref(),
            Some("bad worker \"vllm-0\", want pod=url")
        );
        assert_eq!(
            err("a=http://x,a=http://y").as_deref(),
            Some("worker \"a\" listed twice")
        );
        assert_eq!(err(" , ").as_deref(), Some("no workers configured"));
    }
}
