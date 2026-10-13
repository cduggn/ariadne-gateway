# Gateway design (C11), synthesized 2026-09-28

Status: proposed, awaiting sign-off. It comes from three competing sketches and a cross-judge. The sketches are in
the session scratchpad (`arena-gateway/candidate-{1,2,3}`). Candidate 2 is the base. Candidates 1 and 3 each
contributed several ideas.

## Problem

Ariadne's agent makes 5 to 16 chained, non-streaming chat completions per run. Each call extends a prompt whose
history is cached only on the worker that served the previous step. That caching was worth 95% of prompt tokens
on one worker. The gateway has to do five things:
- make the four ordered decisions (guard, admit, place, queue);
- keep `X-Data-Class: restricted` on the box by construction;
- keep vllm-1 unroutable until it is warm;
- export class-10 `orch_*` metrics, which the dashboards and the presentation depend on;
- forward the body byte for byte.

The client never retries and understands only 429 (tenant) and 503 (capacity). The simulator adds three lessons.
Excluding a stale worker from a 2-worker fleet did more harm than the staleness itself. Priority queueing cut
interactive p99 TTFT 46× against FCFS. Admission and dispatch must not double-book capacity.

## Usage

Code lives in `~/workspace/cluster-doctor/gateway/`, a Go module with its own `go.mod` on go1.27. It uses
`prometheus/client_golang`, plus the roughly 90-line expfmt parser copied from `inference-gateway`. The old
`inference-gateway` repo stays as the course-lab reference and is not edited.

```sh
make gateway-image      # gateway/Dockerfile → docker.io/cdugga/cluster-doctor-gateway (amd64+arm64, public), digest pinned in the manifest
make gateway            # warm-up ConfigMap + deploy/k8s/gateway.yaml applied on the node
make scale N=2          # vllm-1 appears; orch_replica_warm{pod="vllm-1"} stays 0 until two warm probes pass
make tunnel             # TUNNEL ?= svc/gateway, so localhost:8000 reaches the gateway; TUNNEL=pod/vllm-0 is the old direct path
make golden TAG=gw WORKERS=2 CONC=8
make sweep WORKERS=2    # also saves the gateway's /metrics and both pods' /metrics (through /debug/workers/<pod>/metrics)
make gateway POLICY=least_loaded   # control arm for the stickiness A/B
make demo               # laptop only: two fake vLLM processes, the gateway and run_golden; proves the pipeline without a GPU
```

The doctor doesn't change. It keeps `--base-url http://127.0.0.1:8000/v1`. The gateway runs in the cluster as a
one-replica Deployment with a ClusterIP Service on port 8000. Its image is a static binary on distroless, built for
amd64 and arm64 by `gateway/Dockerfile` and pulled by digest from a public Docker Hub repository, so the node needs no
registry credentials. (It first ran from a hostPath binary; that shortcut is ruled out by Pod Security baseline and by
read-only nodes, so the image replaced it.) Its `prometheus.io/scrape`
annotations let the existing Prometheus job scrape it. We rejected running it on the laptop. The scraper would then
reach the pods over SSH, which makes the 2-second staleness line meaningless, and Prometheus couldn't scrape it.

The handler makes three calls:

```go
g := decide.Inspect(body, header, now)          // 400, or a typed Request
d, tk := gate.Admit(ctx, g.Req)                  // decide, reserve and queue under one lock
res := forward(tk.Worker, body); tk.Release(res)  // bytes unchanged; the usage anchors the run's next step
```

## Shape

**Data structures.**
- `decide.Request` holds the parsed run id and step (from `X-Request-Id`), tenant, priority, data class, a
  prompt-token estimate and the maximum output tokens. The enums are total: an unknown priority parses as batch and
  an unknown data class as restricted.
- `decide.Snapshot` is a by-value merge of two single-writer sources. `Scraped` is written by one goroutine per
  worker. `Ledger` holds the gateway's reservations, written only under the Gate lock, and records which requests
  were dispatched after each worker's last scrape. The effective free KV is the scraped free KV minus those
  requests' estimates, so a burst between two scrapes can't pile onto one pod.
- The run table maps a `RunID` to its worker, a boot counter and the last `usage.prompt_tokens`. It is the prefix
  index, so there is no tokenizer and no trie. Warm-up primes the shared prefix on every worker, which means the run
  id alone names the only prefix that differs between workers.

**Pipeline.** Everything is in `decide`, which is pure. The clock and randomness are passed in, and an import test
forbids `net`, `sync` and `os`.
1. `Inspect` validates JSON and size. `stream:true` returns 400, because the doctor never streams.
2. `Pick` keeps Ready workers only and applies the staleness rule below. It then calls `ShouldShed` on each
   candidate, as the simulator's H4 check does, so routing and admission can't disagree. The gates run in order:
   `tenant_tokens` (429), `no_eligible_pod`, `kv_free`, `timeout_queue`, `p99_spread` (batch only).
   - `kv_free` sheds when effective free KV falls below 0.20, the 0.80 line. A run continuing on its own worker is
     exempt down to 0.05, because its history is already resident.
   - When candidates refuse for different reasons, the lowest gate wins, so a 429 always outranks a 503.
   - The policy is `prefix_then_load`: keep the bound worker if its boot counter matches and its load is within the
     best other worker's plus 0.25 load units (`StickSlack`, four of sixteen in-flight slots), otherwise pick the
     least loaded. The bound worker is scored without the run's own history (its last step's `prompt_tokens` as a
     share of the worker's KV pool, clamped at zero), because that KV moves with the run and keeping the run costs
     nothing extra. Before 2026-10-13 it counted, and one 14.5k-token run on an idle 51k pool read 0.28 over the
     other worker, broke its own stickiness and bounced between workers (D-52). `least_loaded` and `p2c` exist for
     the A/B test and for label parity, and score every worker with its full load. At 2 workers `p2c` is the same as
     `least_loaded`.
3. Stay or leave. Only a 503 reaches the overflow decision. `MayLeave` returns an `Offboxable` only for
   non-restricted requests. It has unexported fields, and `overflow.Overflow.Send` and the backend client both call
   `Valid()`, so a zero value refuses too. A fuzz test asserts that no restricted input ever produces an overflow
   route. The overflow backend (`gateway/internal/overflow`, runbook in `design/overflow.md`, D-53) is off unless
   `GW_OVERFLOW=true`: a Superlinked client on the OpenAI chat-completions shape behind a probed circuit, which answers
   the original 503 whenever the backend is unavailable, errors, or is at its in-flight cap. Every saturated
   restricted 503 increments `orch_overflow_total{result="blocked_invariant"}` whatever the switch says, and
   `orch_restricted_offbox_total` is pre-registered at 0 and stays there.
4. Queue. This is the only step that waits, and it lives in `fleet.Gate`. Each worker has an in-flight cap,
   `--max-inflight`, default 16 (the sweep also runs a 32 arm). Past the cap, requests wait in two lanes, and
   interactive drains first. The queue budgets are 10 s for interactive and 30 s for batch, under the client's
   120 s timeout. A request that has been queued is never re-picked. An interactive arrival at a full queue
   displaces the newest batch waiter (`queue_full`), so batch sheds first by construction.

**Staleness per worker.** Each worker ages independently.
- A stale worker (2 to 10 s since its last scrape) stays eligible while the fresh workers left would hold under
  0.75 of Ready capacity. It is then picked on its last-known telemetry and counted on
  `orch_pick_unknown_snapshot_total{pod}`. On a 2-worker fleet a stale worker is therefore never dropped. On 8 workers
  up to two can be dropped. This is the "floor on surviving capacity" from the simulator report.
- A worker with no scrape for 10 s is Down and always excluded. If every worker is Down, requests get a 503, because
  the warm-up requirement forbids routing to a pod we can't see.

**Warm-up gate.** Each worker moves through Down, Warming and Ready, and only Ready workers can be picked. On every
Down-to-Up transition, the worker's own goroutine replays the doctor's recorded step-1 request with
`max_completion_tokens: 1`. Two consecutive probes must come back under 1 s; a cold probe takes about 4.5 s. Probe
times go to `orch_warmup_probe_seconds{pod}`, which is the re-quoted TTFT the brief asks for. A restart is detected
as a Down period followed by recovery, not by `process_start_time_seconds`, which vLLM may not export. A restart
also drops that worker's run bindings. `python -m serving.warmup` writes the recorded body, so the warm-up prefix is
byte-identical to live traffic.

**Quota.** A fixed tenant set. Unknown tenants share one bucket, so a client can't mint fresh bursts by rotating
`X-Tenant`. The burst is at least 32k tokens, the largest context plus output. The quota is charged at admit and
corrected to the real usage afterwards. A refused request is never charged.

**Metrics.** These use the class-10 `orch_*` names:
- `requests`, `shed{reason,code}`, `pick{pod,policy}`, `pick_unknown_snapshot`, `sticky{outcome,reason}`;
- `tokens_in_flight{phase=queued|running}`, `kv_free_ratio`, `overflow{result}`, `restricted_offbox`;
- `completed{pod,status,finish_reason}`, `request_duration_seconds{stage=gateway|pick|queue|local|overflow|e2e}`;
- `replica_{healthy,warm,saturating,kv_free_ratio,tokens_in_flight,waiting,running,queue_depth,active_requests,snapshot_age_seconds}{pool,pod}`,
  computed at scrape time from the gate's own view;
- `prompt_tokens_total{pod,kind=shared_hit|run_hit|miss}`, split from each response's `cached_tokens` with a
  configured shared-prefix length (3,899: prefix plus cluster card).

For per-hop time, each response carries `Server-Timing`, `X-Pod`, `X-Sticky` and `X-Gateway-Queue-Ms` headers, and
the gateway writes one JSON log line per request keyed by `X-Request-Id`. vLLM logs the same id. Finally, a test
checks that every metric name used in `deploy/observability/dashboards/gateway.json` is one the gateway emits.

## Synthesis decision

Candidate 2 (a new minimal gateway with a pure decision core) is the base. The judge scored it 28, against 22 for
candidate 1 (evolve `inference-gateway`) and 15 for candidate 3 (thinnest proxy).
- **Why candidate 2.** It is the only one that closes the admit/dispatch double-booking gap. It encodes the most
  invariants structurally: an import-purity test, total enums, and a proof value plus a fuzz test for the overflow
  rule. It has a three-call handler with no deletion work to do first.
- **Taken from candidate 1:**
  - the capacity-floor staleness rule, which also fixes a fail-open branch in candidate 2 that could never run;
  - the integration scenarios (replaying the simulator's T3 stale-worker case, a rolling restart with re-warm,
    sha256 byte identity on recorded golden bodies);
  - `make demo`, `orch_restricted_offbox_total`, and the per-request response headers.
- **Taken from candidate 3:**
  - a fixed tenant set with a shared bucket for unknown tenants;
  - returning 400 on streaming, which removes the SSE relay;
  - `client_golang` for the metrics;
  - the `blocked_invariant` label wording.
- **Rejected:**
  - Candidate 1's interface-based proof. Go allows embedding the interface, so another package could forge it.
  - Candidate 1 deriving the shared-prefix length from warm-up. The warm-up request also contains the card and the
    task, so the length would be too long.
  - Candidate 3's lack of a gateway queue, which breaks the priority MUST.
  - Candidate 3's hash placement. A run that spills to the other worker flips back and re-prefills its history.
  - Candidate 3's permanent Down state, with no way back to Ready.
  - Detecting restarts through `process_start_time_seconds`, which is version-sensitive.

## Tradeoffs accepted

- We accept a tokenizer-free estimate of the prompt size: bytes ÷ 3.5 on a run's first step, then the previous step's
  real usage. In exchange there is no tokenizer. Errors only shift margins, and the scraped KV corrects them.
- We accept one mutex around deciding and reserving, in exchange for no double-booking. At 32 concurrent requests it
  is not contended.
- We accept an in-flight cap of 16, below `max-num-seqs` 32, so that priority ordering happens at the gateway. The
  32 arm shows what that costs.
- We accept a static two-worker list, in exchange for no discovery code. The sliced topology can't exceed two anyway.
- We accept no upstream retry and no re-pick. A pod that dies mid-request shows up as a 502.

## KV hop (opt-in)

When the gateway moves a run off the worker that holds its history (`broken_load` or `broken_shed`), the new worker
normally recomputes that history. With the hop on, the gateway copies the KV instead, through vLLM's
`MooncakeConnector`. The gateway's flag defaults to off (`GW_HOP=false`), and `make up` turns it on for an H100 node.
For today's workload a recompute takes a second or two, so the hop matters only if contexts grow. The code is in
`gateway/internal/hop`, separate from the routing core.

**Decision (pure, `hop.Config.Decide`).** Hop only when both hold:
1. the run's history (its last step's `usage.prompt_tokens`) is at least `GW_HOP_MIN_TOKENS` (default 8,192);
2. `overhead + tokens × KV bytes/token ÷ transfer bandwidth` < `tokens ÷ prefill rate`, where `tokens` is the history
   beyond the shared prefix every worker already holds from warm-up.

| Setting | Default | Where it comes from |
|---|---|---|
| `GW_HOP_MIN_TOKENS` | 8,192 | Below this, a recompute is under ~1.2 s on a slice and not worth two extra round trips |
| `GW_HOP_KV_BYTES_PER_TOKEN` | 147,456 (8B) | `make gateway` sets it from `serving.fit --gateway-env` for the deployed model |
| `GW_HOP_PREFILL_TOKENS_PER_S` | 3,810 (8B slice) | Same, from the fit's prefill estimate; replace with a measured rate |
| `GW_HOP_TRANSFER_BYTES_PER_S` | 2e9 | A guess for TCP between two pods on one host; **measure before trusting the rule** |
| `GW_HOP_OVERHEAD`, `GW_HOP_TIMEOUT` | 50 ms, 10 s | The fixed cost of a hop; the longest wait for a busy source before recomputing |
| `GW_HOP_RECEIVE_TIMEOUT` | 30 s | The longest wait for the destination to answer a hopped request; past it the request goes again without the hop (D-52) |
| `GW_HOP_MAX_INFLIGHT` | 4 | Hops running at once; past it a moved run recomputes (`busy`) |

**Transport (`hop.Mooncake`),** as vLLM v0.29.0's own Mooncake proxy does it:
1. Find the source's engine id from its bootstrap registry, `GET http://<pod>:8998/query`. The id is cached per
   worker and dropped when the hold fails, when the fleet sees the worker leave Ready or restart (`Fleet.OnNotReady`
   → `Hopper.Forget`), and when a destination fails to pull from it, because a restarted worker has a new id and a
   hold against the old one succeeds while the pull hangs (D-52).
2. Send the source the request with `kv_transfer_params={do_remote_decode, transfer_id}`, `max_tokens=1` and no
   streaming. The source prefills from its own prefix cache and holds the blocks. It holds them only if it stops at the
   length cap, so the gateway checks `finish_reason == "length"`.
3. Forward the request to the destination with `{do_remote_prefill, remote_engine_id, remote_bootstrap_addr,
   transfer_id}`. vLLM pulls only the blocks the destination doesn't already have.

**Guarantees and limits.**
- Any failure (registry, source error, timeout, a source that didn't hold) forwards the original body, and the
  destination recomputes. A hop can't fail a request.
- The destination's answer to a hopped request is bounded by `GW_HOP_RECEIVE_TIMEOUT` (30 s), not the 115 s upstream
  timeout. If it gives no answer in time, or a 5xx, the gateway forgets the source's engine id, counts the hop as
  `failed`, and sends the request once more without `kv_transfer_params` under what is left of the upstream timeout,
  so the destination recomputes. The second attempt carries `X-Request-Id` with a `-recompute` suffix, because vLLM
  refuses a request id it still holds. The log line says `hop_fallback=true`. A 4xx is relayed as it is.
- Both workers are the fleet's own pods, so a hop never sends data off the box.
- Each hop's `transfer_id` is 128 random bits, never derived from the client's `X-Request-Id`. The id names the held
  blocks on the source, so one tenant can't name, or collide with, a transfer made for another.
- At most `GW_HOP_MAX_INFLIGHT` hops run at once. Each sends the source a request outside admission and may pin its
  blocks, so a burst of moves can't become a burst of hops.
- Only the gateway sets `kv_transfer_params`. The guard refuses it from a client with a 400 (`kv_transfer_params`),
  whether or not the hop is on, because a worker running the connector would otherwise connect to any
  `remote_bootstrap_addr` a client names (SSRF) or pull another request's KV by its `transfer_id`.
- With `HOP=1` the render adds a NetworkPolicy (`vllm-kv-hop`). Workers accept the API port only from the gateway and
  Prometheus, the bootstrap port only from the gateway, and anything only from another worker. So nothing in the
  cluster can bypass the gateway's guard by calling a worker directly. `make tunnel` uses `kubectl port-forward`, which
  enters the pod directly and is unaffected.
- A hop doesn't relieve the source's KV pressure. The blocks stay until vLLM evicts them. A hop whose destination
  request never arrives pins the source's blocks for `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` (480 s).
- The workers must run the connector. `make deploy HOP=1` adds `--kv-transfer-config` (`kv_both`, `tcp` because the
  node has no RDMA) and port 8998, from `serving.json` `kv_hop`.
- On hardware, 53 hops completed the protocol with none failed: 8 on 2026-10-07 and 45 on 2026-10-08 (findings F35,
  F45). Still unverified are the copy
  bandwidth between two HAMi slices on one GPU, and whether a destination really pulls Qwen3.8's hybrid state rather
  than recomputing it.

**Signals.** `orch_hop_total{result=hopped|failed|busy|below_threshold|recompute_cheaper}`, the `hop` stage of
`orch_request_duration_seconds`, an `X-Hop` response header, and `hop`/`hop_ms`/`hop_fallback` on the request log
line (`lab/tail.py` shows the hop column, with `*` for a fallback). A successful hop shows on the destination as
`cached_tokens ≈ prompt_tokens` for that step. A `failed` with `hop_fallback=true` means the destination could not
pull and recomputed on the second attempt.

## Open questions

- Tenant rates: `platform` 200k tokens/min with a 32k burst, other tenants shared at a lower rate. Does that make the
  429 path demonstrable without firing during a sweep?
- Does `lam push` handle a single binary? If not, `scp` through `lam env` works.

## Next step

Put the `decide` package and its table tests (the T3 case and the restricted fuzz test first) into
`gateway/internal/decide`, from the candidate 2 sketch plus the grafts above.
