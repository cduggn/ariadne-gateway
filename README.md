# ariadne-gateway

An inference gateway for a small fleet of vLLM workers, written in Rust. It sits in front of the workers and makes
four decisions for every chat completion, in order. It guards the request, admits it against a tenant token budget,
places it on a worker, and queues it when every worker is busy. Then it forwards the body byte for byte.

This is a port of the Go gateway in [cduggn/Ariadne](https://github.com/cduggn/Ariadne) (`gateway/`). It is a
drop-in replacement: the same flags and `GW_*` environment variables, routes, status codes, headers, JSON request
log, and `orch_*` metrics. The Grafana dashboard and the lab scripts written for the Go binary work unchanged.

## What the gateway does

- **Keeps a run on its worker.** An agent run makes 5 to 16 chained calls, and each call extends a prompt whose
  history is cached only on the worker that served the previous step. The `prefix_then_load` policy keeps the run
  there unless that worker is more than 0.25 load units busier than the best other one.
- **Sheds before it overloads.** A request is refused with 429 when its tenant's bucket cannot cover it, and with 503
  when no worker has free KV cache, its queue wait would pass the deadline, or (for batch requests) the worker's p99
  time to first token has drifted far from its p50.
- **Puts interactive traffic first.** Past each worker's in-flight cap, requests wait in two lanes. Interactive
  drains first, and an interactive arrival at a full queue displaces the newest batch waiter.
- **Keeps restricted data on the box.** A 503 may go to an off-box overflow backend only when the request's
  `X-Data-Class` is `internal` or `public`. The overflow client accepts only a permit that the stay-or-leave rule
  minted, so no other code path can send a restricted request off the box.
- **Warms a worker before routing to it.** A worker becomes routable only after two warm-up probes in a row answer
  in under 1 s.
- **Moves KV instead of recomputing it** (off by default). When a run must move to another worker, the gateway can
  copy its KV cache through vLLM's `MooncakeConnector`.

[`design/gateway.md`](design/gateway.md) explains each rule and the tradeoffs behind it.
[`design/overflow.md`](design/overflow.md) is the overflow runbook. Both were written for the Go implementation. The
mapping from its packages to this repository is in [Layout](#layout).

## Build and run

You need Rust 1.98 or later.

```sh
cargo build --release
./target/release/gateway \
  --warm-body warm.json \
  --workers vllm-0=http://vllm-0:8000,vllm-1=http://vllm-1:8000
```

`--warm-body` is the recorded first request of an agent run. Warm-up probes replay it with
`max_completion_tokens: 1`. Every flag also reads a default from `GW_<NAME>`, with dashes turned into underscores
(`--max-inflight` reads `GW_MAX_INFLIGHT`). Run `gateway -h` for the full list.

To try it on a laptop without a GPU, start two fake workers and point the gateway at them:

```sh
./target/release/fakevllm --listen 127.0.0.1:18001 &
./target/release/fakevllm --listen 127.0.0.1:18002 &
echo '{"model":"m","messages":[{"role":"user","content":"warm-up"}],"max_completion_tokens":1}' > warm.json
./target/release/gateway --listen 127.0.0.1:18000 --warm-body warm.json \
  --workers vllm-0=http://127.0.0.1:18001,vllm-1=http://127.0.0.1:18002
curl -s localhost:18000/debug/workers
```

## Deploy

`deploy/k8s/gateway.yaml` runs one replica behind a ClusterIP Service on port 8000. Keep it at one replica: the run
table and the queue live in memory.

1. Build and push the image: `make image IMAGE=docker.io/<you>/ariadne-gateway TAG=<tag>`. It builds `linux/amd64`
   and `linux/arm64` as a static binary on distroless.
2. Replace the `image:` line in `deploy/k8s/gateway.yaml` with the pushed image, pinned by digest.
3. Create the warm-up ConfigMap: `kubectl create configmap gateway-warmup --from-file=warm.json`.
4. Apply the manifest: `kubectl apply -f deploy/k8s/gateway.yaml`.

Prometheus scrapes `/metrics` through the pod annotations. `deploy/dashboards/gateway.json` is the Grafana
dashboard, and a test fails if it names a metric the gateway does not emit.

## Verify a change

```sh
make check    # cargo fmt, clippy with -D warnings, and the tests
make parity   # the same scripted traffic through the Go gateway and this one
```

`make parity` clones Ariadne at a pinned commit, builds the Go gateway and its fakes, and runs
`parity/parity.py` against both binaries. The script sends a fixed set of requests: valid runs, every guard
refusal, saturated workers with and without an overflow backend, each pick policy, and bad startup flags. It
records what a client, Prometheus and a log reader see, with timings blanked. The run fails on any difference. CI
runs both targets on every push.

## Differences from the Go gateway

These are deliberate. Each one closes a gap the Go code had, and none changes how a well-formed request is handled.

- The guard refuses `stream` unless it is absent, `null` or `false`. Go refused only `true`, but vLLM also reads
  `1` and `"true"` as true.
- The guard refuses a body with invalid UTF-8 in a string or nesting deeper than 128 levels (`bad_json`). Go
  accepted both.
- `X-App` passes through the same filter as `X-Tenant` before anything uses it.
- `max_completion_tokens` is capped at 2^32 inside the gateway, so token sums cannot overflow. A huge value is
  still refused with 429 `tenant_tokens`, as in Go.
- Numbers scraped from a worker's `/metrics` are validated. A negative or non-finite KV usage reads as full, and a
  sample line with more than 64 labels is a parse error.
- A body that the overflow client cannot rewrite fails only its own request. In Go it opened the circuit for every
  tenant.
- A hop rate of `NaN` is refused at startup.
- On SIGTERM the gateway drains in-flight requests for up to 30 s. The Go binary logged "gateway stopped" and exited
  without waiting.
- `/metrics` has no `go_*`, `process_*` or `promhttp_*` families.
- The server has no 125 s write timeout. The 10 s header read timeout is kept.
- A client that leaves while its request waits in the queue gets a log line without the placement fields.

## Performance

These numbers come from one run of `ab` on an Apple silicon laptop, through two zero-latency fake workers, for
1,500 requests. Every request was a forwarded 200.

| | Go | Rust |
|---|---|---|
| Requests per second, 1 client | 10,321 | 11,292 |
| Requests per second, 16 clients | 16,775 | 42,109 |
| Mean time per request, 16 clients | 0.95 ms | 0.38 ms |
| Resident memory under load | 21 to 26 MB | 10 to 12 MB |
| Release binary | 14.6 MB | 5.2 MB |

A real vLLM step takes seconds, so the gateway's own time is a small share of end-to-end latency with either
binary. The difference matters for memory on a shared node and for headroom under bursts.

## Layout

| Path | Contents | Go package |
|---|---|---|
| `crates/decide` | The pure decision core: guard, shed rules, pick, stay or leave. No I/O, no clock. | `internal/decide` |
| `crates/gateway/src/fleet` | The gate (decide and reserve under one lock), the queue, the worker loop, the `/metrics` parser | `internal/fleet` |
| `crates/gateway/src/hop` | The KV hop cost rule and the Mooncake transport | `internal/hop` |
| `crates/gateway/src/overflow` | The overflow circuit and the Superlinked client | `internal/overflow` |
| `crates/gateway/src/metrics` | The `orch_*` exposition | `internal/metrics` |
| `crates/gateway/src/serve` | The HTTP layer and the per-request event | `internal/serve` |
| `crates/gateway/src/fake` | Fake vLLM and overflow backends for tests and the laptop setup | `internal/fakevllm`, `internal/fakeoverflow` |
| `crates/gateway/src/bin` | `gateway`, `fakevllm`, `fakeoverflow` | `cmd/*` |
| `parity/parity.py` | The Go-versus-Rust comparison | |
