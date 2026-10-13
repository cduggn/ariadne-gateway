# Overflow to Superlinked: runbook (D-53)

Status: built and tested against a fake backend, off in the deployment, waiting for Superlinked API access (F22).
The gateway's stay-or-leave rule (`design/gateway.md`, pipeline step 3) is unchanged: only a 503 may leave the box, and
a restricted request never does. This page is what to do when the key arrives, what to watch, and how to prove the
path works without Superlinked.

## What it does

When the gateway cannot serve a request (a 503: `kv_free`, `timeout_queue`, `queue_full`, `no_eligible_pod`,
`p99_spread`) and `decide.MayLeave` lets it leave, the serve layer offers it to `overflow.Overflow`, the circuit over
one backend (`gateway/internal/overflow`):

1. The permit is checked again (`Offboxable.Valid`), and once more inside the backend client, so nothing restricted can
   be sent even by a caller that skipped the rule.
2. If the circuit is open (`unavailable`) or `GW_OVERFLOW_MAX_INFLIGHT` calls are already out (`capped`), the client
   gets the 503 it would have had anyway, byte for byte.
3. Otherwise the body is posted to the backend's chat-completions path with the backend's bearer key, the model name
   rewritten to `GW_OVERFLOW_MODEL` when set, and none of the client's headers (no run, tenant or request id leaves).
4. A 200 is relayed unchanged (`sent`), with `X-Overflow: sent`, `X-Overflow-Backend`, `X-Gateway-Reason` (why the
   gateway could not serve it) and `Server-Timing: overflow;dur=`. No `X-Pod`, because no worker served it.
5. Anything else is `failed` and the client gets the original 503. When the fault is the backend's (no answer, a
   timeout, 401/402/403, 429 or a 5xx) the circuit opens until the next probe passes, so a dead or out-of-credit backend
   costs at most one call per probe interval. A 4xx about the one request (400, 413, 422) fails only that request: one
   tenant's bad body cannot switch overflow off for everyone else.

The circuit starts open. A probe (`GET` the probe path with the key; any 2xx passes) runs at startup and every
`GW_OVERFLOW_PROBE_INTERVAL`. A failed probe never fails the gateway: it logs, keeps `orch_overflow_backend_available`
at 0, and every 503 stays a 503.

Before a request leaves, `Gate.ChargeOffbox` charges the tenant's bucket with the same estimate admission uses and
`SettleOffbox` corrects it to the backend's reported usage afterwards, or refunds it when the backend did not answer. A
bucket that cannot cover the estimate keeps the 503 on the box with result `quota`. Off-box answers therefore spend
quota exactly like local ones, so a tenant cannot spend the backend's tokens without limit whenever the box is full.

Not done, on purpose: a run that overflowed is not pinned to the backend. Its next step goes through admission again
and lands on a worker if one has room (the history recomputes there), or overflows again.

## Configuration

Every flag reads `GW_<NAME>`. The deployment (`deploy/k8s/gateway.yaml`) sets them all, with overflow off.

| Flag / env | Default | Meaning |
|---|---|---|
| `--overflow` / `GW_OVERFLOW` | `false` | The switch. Off means today's behaviour exactly (`no_backend`). |
| `--overflow-base-url` / `GW_OVERFLOW_BASE_URL` | `https://api.superlinked.com` | Origin only; a trailing `/v1` is tolerated and dropped. |
| `--overflow-chat-path` / `GW_OVERFLOW_CHAT_PATH` | `/v1/chat/completions` | Where the body is posted. |
| `--overflow-probe-path` / `GW_OVERFLOW_PROBE_PATH` | `/v1/models` | What the probe fetches. |
| `--overflow-model` / `GW_OVERFLOW_MODEL` | `""` (deployment: `Qwen/Qwen3.8-27B-FP8`) | Replaces the request's `model`; empty forwards the client's. |
| `--overflow-api-key-env` / `GW_OVERFLOW_API_KEY_ENV` | `GW_OVERFLOW_API_KEY` | The variable the key is read from. The key itself is never logged. |
| `--overflow-timeout` / `GW_OVERFLOW_TIMEOUT` | `60s` | One backend call, under the client's 120 s. |
| `--overflow-probe-interval` / `GW_OVERFLOW_PROBE_INTERVAL` | `30s` | Re-probe period; also how long the circuit stays open after a failure. |
| `--overflow-max-inflight` / `GW_OVERFLOW_MAX_INFLIGHT` | `4` | Concurrent backend calls; past it a 503 stays a 503 (`capped`). |

The key comes from the Kubernetes Secret **`gateway-overflow`**, key **`api-key`**, through a `secretKeyRef` marked
`optional`, so the pod starts without it. There is no Secret in the repo and there must never be one.

**Confirmed from Superlinked's public docs** (superlinked.com/docs, HTTP API reference, 2026-10-13): their engine's
gateway serves `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses` and `POST /v1/embeddings` as
OpenAI-compatible routes, plus `GET /v1/models` and `/readyz`. **Assumed, not confirmed:** that the hosted API at
`api.superlinked.com` has the same paths (the course lab's `class10/router/overflow.py` posts to
`https://api.superlinked.com/v1/chat/completions`, and F22 got 402s from it, so the host exists), that it takes
`Authorization: Bearer <key>` (the course lab does this; the public docs describe no auth scheme), that it accepts
`tools`/`tool_choice` (not documented), and the served model name (F22 saw `Qwen/Qwen3.8-27B-FP8`). All four are
flags, so a different answer is a `kubectl set env`, not a rebuild.

## Enabling it when the key arrives

1. Check the facts above against the real API from the laptop, with the key in the environment, never on the command
   line or in a file under the repo:
   ```sh
   curl -s -H "Authorization: Bearer $KEY" https://api.superlinked.com/v1/models | head -c 600
   ```
   If the models list is a 2xx, the probe will pass. Note the model id it serves and whether it matches
   `GW_OVERFLOW_MODEL`. If the paths differ, set `GW_OVERFLOW_CHAT_PATH` / `GW_OVERFLOW_PROBE_PATH`.
2. Create the Secret on the cluster (never commit it):
   ```sh
   lam ssh cluster-doctor -- kubectl create secret generic gateway-overflow --from-literal=api-key="$KEY"
   ```
3. Turn it on: `make gateway OVERFLOW=1` (sets `GW_OVERFLOW=true` and restarts), or
   `kubectl set env deployment/gateway GW_OVERFLOW=true GW_OVERFLOW_MODEL=<id from step 1>`.
4. Read the startup lines: `overflow on` (with `key_set=true`) then `overflow backend available`. If instead you see
   `overflow backend still unavailable` with a 401/402/403, the key or the account is the problem and the gateway is
   serving exactly as before.
5. Make it fire once: with the tunnel up, send an `X-Data-Class: internal` request while both workers are above the
   KV line (a sweep at 32 does this), or temporarily `kubectl set env deployment/gateway GW_MAX_INFLIGHT=1` and run
   two. Expect a 200 with `X-Overflow: sent`, and `orch_overflow_total{result="sent"}` to move.

To turn it off: `make gateway` (OVERFLOW defaults to 0) or `kubectl set env deployment/gateway GW_OVERFLOW=false`.
Deleting the Secret alone does not stop it until the next restart; the key is read at startup.

## What to watch

- `orch_overflow_backend_available` (gauge): 1 while the last probe passed and no send failed since. On the dashboard
  this is the first thing to look at when `sent` stops moving.
- `orch_overflow_total{result}`: `sent` is the win; `failed` with the backend available means the chat path or the
  model is wrong (read the `overflow failed` warning, it carries the status); `unavailable` is the circuit doing its job;
  `capped` means raise `GW_OVERFLOW_MAX_INFLIGHT` or accept it; `blocked_invariant` is restricted traffic saturating,
  which overflow cannot help; `no_backend` means the switch is off.
- `orch_restricted_offbox_total` must stay 0. The `RestrictedRequestOffBox` alert (critical) fires if it ever moves;
  it is now incremented by the metrics layer if an answer marked `sent` carries a restricted class, which nothing in the
  gateway can produce (`MayLeave`, the permit, the client's own check, and the fuzz test).
- `orch_request_duration_seconds{stage="overflow"}`: the backend's latency. The timeout bounds it at 60 s.
- `orch_completed_total{pod="overflow"}`: answers served off the box, next to the per-worker series.
- The request log: `overflow` and `overflow_ms` on every line; `lab/tail.py` shows the `overflow` column.
- Spend: every `sent` and every `failed` is a paid call. `GW_OVERFLOW_MAX_INFLIGHT` bounds the rate, not the total.

## Verifying without Superlinked

The fake backend (`gateway/internal/fakeoverflow`, `gateway/cmd/fakeoverflow`) serves the two routes the client
touches, checks a bearer key, and records what it saw.

- Unit and integration tests, no network:
  `cd gateway && go test ./internal/overflow/ ./internal/serve/ -run 'Overflow|Restricted503|Unavailable|BackendError'`.
  `serve` proves through the HTTP path that a non-restricted 503 is served by the fake with `sent` logged, that a
  restricted 503 under every spelling of the header never reaches the fake, and that an unavailable backend returns a
  503 identical to the control gateway's.
- Laptop demo: `make demo OVERFLOW=1`. Both fake workers report a full KV cache, a fake backend plays Superlinked on
  `127.0.0.1:18010` with key `demo-key`, and the script sends one restricted and one internal request. The restricted
  one is a 503 with `Retry-After` and no `X-Overflow`; the internal one is a 200 with `X-Overflow: sent`. It then
  prints the log through `lab/tail.py` and the `orch_overflow_total` series.
- Against a real but misbehaving backend, `fakeoverflow --status 402` plays the F22 case (the probe passes, every chat
  fails, the circuit opens), and `--down` plays an outage (the probe fails, nothing is sent).
