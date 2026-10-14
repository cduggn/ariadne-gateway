#!/usr/bin/env python3
"""Drive one gateway build through a fixed script and print what a client, Prometheus and a log reader observe.

    parity.py observe --gateway BIN --fakevllm BIN --fakeoverflow BIN --out FILE [--port-base N]
    parity.py diff A.json B.json

observe starts two fake vLLM workers, the gateway and (for the saturation scenes) a fake overflow backend, sends the
script, and writes normalized observations as JSON. Timings are replaced by a placeholder, so two builds that behave
the same write the same file. diff prints every difference and exits 1 when there is one.
"""
import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import time

WARM = {"model": "Qwen/Qwen3-8B-AWQ", "messages": [{"role": "user", "content": "warm-up"}], "max_completion_tokens": 1}
KEPT_HEADERS = {
    "content-type", "retry-after", "x-pod", "x-policy", "x-sticky", "x-hop", "x-gateway-queue-ms", "server-timing",
    "x-overflow", "x-overflow-backend", "x-gateway-reason", "x-content-type-options", "allow",
}
TIMING = re.compile(r"\d+(\.\d+)?")
VOLATILE_LOG = {"time", "queue_ms", "upstream_ms", "total_ms", "overflow_ms", "hop_ms"}
VOLATILE_METRIC = re.compile(
    r"_seconds(_sum|_bucket)?$|^orch_replica_snapshot_age_seconds$|^(go|process|promhttp)_|tokens_in_flight|_warmup_"
)


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    kept = {}
    for k, v in resp.getheaders():
        k = k.lower()
        if k in KEPT_HEADERS:
            kept[k] = TIMING.sub("N", v) if k in ("server-timing", "x-gateway-queue-ms") else v
    return resp.status, kept, raw


def normalize_body(raw):
    try:
        doc = json.loads(raw)
    except ValueError:
        return {"text": raw.decode("utf-8", "replace").strip()}
    return {"json": scrub(doc)}


def scrub(doc):
    """Replace ids and timestamps the fakes mint per response."""
    if isinstance(doc, dict):
        return {k: ("<volatile>" if k in ("id", "created") else scrub(v)) for k, v in doc.items()}
    if isinstance(doc, list):
        return [scrub(v) for v in doc]
    return doc


def chat(port, request_id, body=None, **headers):
    h = {"Content-Type": "application/json", "X-Request-Id": request_id}
    for k, v in headers.items():
        h["X-" + k.replace("_", "-").title()] = v
    payload = body if body is not None else json.dumps(WARM).encode()
    status, kept, raw = call(port, "POST", "/v1/chat/completions", payload, h)
    return {"request_id": request_id, "status": status, "headers": kept, "body": normalize_body(raw)}


def metrics(port):
    """Parse the exposition into families: type, help, and series with their values (volatile ones blanked)."""
    _, _, raw = call(port, "GET", "/metrics")
    families, series = {}, {}
    for line in raw.decode().splitlines():
        if line.startswith("# HELP "):
            name, _, text = line[7:].partition(" ")
            families.setdefault(name, {})["help"] = text
        elif line.startswith("# TYPE "):
            name, _, kind = line[7:].partition(" ")
            families.setdefault(name, {})["type"] = kind
        elif line and not line.startswith("#"):
            key, _, value = line.rpartition(" ")
            name = key.split("{", 1)[0]
            if name.startswith(("go_", "process_", "promhttp_")):
                continue
            series[key] = "N" if VOLATILE_METRIC.search(name) else value
    families = {k: v for k, v in families.items() if not k.startswith(("go_", "process_", "promhttp_"))}
    return {"families": families, "series": series}


def request_logs(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("msg") != "request":
                continue
            out.append({k: ("N" if k in VOLATILE_LOG else v) for k, v in rec.items()})
    return out


def log_shape(path):
    """Every distinct (level, msg, sorted keys) the gateway logged, so non-request lines are compared by shape."""
    shapes = set()
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                shapes.add(("unparsed", line.strip()[:80], ""))
                continue
            shapes.add((rec.get("level", ""), rec.get("msg", ""), ",".join(sorted(rec))))
    return sorted(list(s) for s in shapes)


class Stack:
    def __init__(self, args, workdir, base, kv, overflow, env=None, extra=()):
        self.procs, self.workdir, self.port = [], workdir, base
        warm = os.path.join(workdir, "warm.json")
        with open(warm, "w") as f:
            json.dump(WARM, f, separators=(",", ":"))
        self.log = os.path.join(workdir, f"gateway-{base}.log")
        flags = []
        environ = dict(os.environ, **(env or {}))
        if overflow:
            self.spawn([args.fakeoverflow, "--listen", f"127.0.0.1:{base + 10}", "--key", "demo-key", "--latency", "20ms"])
            environ["GW_OVERFLOW_API_KEY"] = "demo-key"
            flags = ["--overflow", "--overflow-base-url", f"http://127.0.0.1:{base + 10}", "--overflow-model", "demo-model",
                     "--overflow-probe-interval", "1s"]
        for i in (1, 2):
            self.spawn([args.fakevllm, "--listen", f"127.0.0.1:{base + i}", "--kv-usage", str(kv), "--latency", "20ms"])
        workers = f"vllm-0=http://127.0.0.1:{base + 1},vllm-1=http://127.0.0.1:{base + 2}"
        with open(self.log, "w") as out:
            self.spawn([args.gateway, "--listen", f"127.0.0.1:{base}", "--warm-body", warm, "--workers", workers,
                        *flags, *extra], stdout=out, stderr=subprocess.STDOUT, env=environ)
        self.wait_ready()

    def spawn(self, argv, **kw):
        kw.setdefault("stdout", subprocess.DEVNULL)
        kw.setdefault("stderr", subprocess.DEVNULL)
        self.procs.append(subprocess.Popen(argv, **kw))

    def wait_ready(self):
        for _ in range(150):
            try:
                _, _, raw = call(self.port, "GET", "/debug/workers")
                if all(w["phase"] == "ready" for w in json.loads(raw)):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        raise SystemExit(f"gateway on {self.port} never had both workers ready; see {self.log}")

    def stop(self):
        for p in reversed(self.procs):
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def plain(port, method, path, body=None):
    status, kept, raw = call(port, method, path, body)
    return {"call": f"{method} {path}", "status": status, "headers": kept, "body": normalize_body(raw)}


def debug_workers(port):
    _, _, raw = call(port, "GET", "/debug/workers")
    doc = json.loads(raw)

    def shape(v):
        if isinstance(v, dict):
            return {k: shape(x) for k, x in sorted(v.items())}
        if isinstance(v, list):
            return [shape(x) for x in v]
        if isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\dT", v):
            return "timestamp"
        return v if isinstance(v, (str, bool)) else "number"
    return shape(doc)


def scene_normal(args, workdir, base):
    s = Stack(args, workdir, base, kv=0.1, overflow=False)
    p = s.port
    try:
        obs = {"probes": [
            plain(p, "GET", "/healthz"), plain(p, "GET", "/readyz"), plain(p, "GET", "/nope"),
            plain(p, "GET", "/v1/chat/completions"), plain(p, "GET", "/debug/workers/nope/metrics"),
            plain(p, "POST", "/healthz"),
        ]}
        obs["debug_workers"] = debug_workers(p)
        status, kept, raw = call(p, "GET", "/debug/workers/vllm-0/metrics")
        obs["worker_metrics"] = {"status": status, "headers": kept, "body": raw.decode()}
        big = json.dumps({"model": "m", "messages": [{"role": "user", "content": "x" * (1 << 20)}]}).encode()
        obs["guard"] = [
            chat(p, "bad-json-s1", b"{not json"),
            chat(p, "bad-empty-s1", b""),
            chat(p, "bad-array-s1", b"[1,2]"),
            chat(p, "bad-stream-s1", json.dumps(dict(WARM, stream=True)).encode()),
            chat(p, "bad-kv-s1", json.dumps(dict(WARM, kv_transfer_params={"transfer_id": "x"})).encode()),
            chat(p, "bad-big-s1", big),
            chat(p, "bad-nomsg-s1", b'{"model":"m"}'),
            chat(p, "bad-maxout-s1", json.dumps(dict(WARM, max_completion_tokens=-5)).encode()),
            chat(p, "bad-maxout2-s1", json.dumps(dict(WARM, max_completion_tokens=10 ** 7)).encode()),
        ]
        runs = []
        for step in range(1, 6):
            msgs = [{"role": "user", "content": f"turn {i} " + "history " * 40} for i in range(step)]
            body = json.dumps({"model": "m", "messages": msgs, "max_completion_tokens": 64}).encode()
            runs.append(chat(p, f"run-a-s{step}", body, tenant="platform", app="cluster-doctor", priority="interactive",
                             data_class="restricted"))
        runs.append(chat(p, "run-b-s1", tenant="platform", priority="batch", data_class="internal"))
        runs.append(chat(p, "run-c-s1", tenant="somebody-else", priority="interactive", data_class="public"))
        runs.append(chat(p, "run-d-s1", tenant="Bad Tenant!!", priority="weird", data_class="weird"))
        runs.append(chat(p, "no-suffix"))
        runs.append(chat(p, ""))
        runs.append(chat(p, "run-e-s2", tenant="x" * 200))
        obs["runs"] = runs
        obs["metrics"] = metrics(p)
    finally:
        s.stop()
    obs["request_log"] = request_logs(s.log)
    obs["log_shapes"] = log_shape(s.log)
    return obs


def scene_saturated(args, workdir, base, overflow):
    s = Stack(args, workdir, base, kv=0.95, overflow=overflow)
    p = s.port
    try:
        if overflow:
            for _ in range(50):
                with open(s.log) as f:
                    if "overflow backend available" in f.read():
                        break
                time.sleep(0.2)
        obs = {"runs": [
            chat(p, "sat-restricted-s1", tenant="platform", priority="interactive", data_class="restricted"),
            chat(p, "sat-internal-s1", tenant="platform", priority="interactive", data_class="internal"),
            chat(p, "sat-public-s1", tenant="platform", priority="batch", data_class="public"),
            chat(p, "sat-unknown-s1", tenant="platform", priority="batch", data_class="???"),
        ]}
        obs["metrics"] = metrics(p)
    finally:
        s.stop()
    obs["request_log"] = request_logs(s.log)
    obs["log_shapes"] = log_shape(s.log)
    return obs


def scene_policy(args, workdir, base, policy):
    s = Stack(args, workdir, base, kv=0.1, overflow=False, env={"GW_POLICY": policy, "GW_POOL": "h100-half",
                                                                  "GW_MAX_INFLIGHT": "2", "GW_SHARED_PREFIX_TOKENS": "10"})
    p = s.port
    try:
        obs = {"runs": [chat(p, f"pol-{r}-s{n}", tenant="platform") for r in "ab" for n in (1, 2, 3)]}
        obs["metrics"] = metrics(p)
    finally:
        s.stop()
    obs["request_log"] = request_logs(s.log)
    return obs


def scene_startup_errors(args, workdir):
    """What the binary prints and exits with when its configuration is wrong."""
    warm = os.path.join(workdir, "warm.json")
    cases = {
        "no_warm_body": ([], {}),
        "bad_policy": (["--warm-body", warm, "--policy", "nope"], {}),
        "bad_workers": (["--warm-body", warm, "--workers", "vllm-0"], {}),
        "dup_workers": (["--warm-body", warm, "--workers", "a=http://x,a=http://y"], {}),
        "bad_env_int": (["--warm-body", warm], {"GW_MAX_INFLIGHT": "many"}),
        "bad_env_duration": (["--warm-body", warm], {"GW_HOP_TIMEOUT": "soon"}),
        "bad_env_bool": (["--warm-body", warm], {"GW_HOP": "maybe"}),
        "missing_warm_file": (["--warm-body", os.path.join(workdir, "absent.json")], {}),
        "bad_log_level": (["--warm-body", warm, "--log-level", "loud"], {}),
    }
    out = {}
    for name, (argv, env) in cases.items():
        environ = {k: v for k, v in os.environ.items() if not k.startswith("GW_")}
        environ.update(env)
        r = subprocess.run([args.gateway, "--listen", "127.0.0.1:0", *argv], capture_output=True, text=True, env=environ,
                           timeout=20)
        out[name] = {"exit": r.returncode, "stderr": r.stderr.strip().replace(workdir, "<dir>")}
    return out


def observe(args):
    base = args.port_base
    with tempfile.TemporaryDirectory() as workdir:
        with open(os.path.join(workdir, "warm.json"), "w") as f:
            json.dump(WARM, f)
        obs = {
            "normal": scene_normal(args, workdir, base),
            "saturated_no_backend": scene_saturated(args, workdir, base + 20, overflow=False),
            "saturated_overflow": scene_saturated(args, workdir, base + 40, overflow=True),
            "least_loaded": scene_policy(args, workdir, base + 60, "least_loaded"),
            "p2c": scene_policy(args, workdir, base + 80, "p2c"),
            "startup_errors": scene_startup_errors(args, workdir),
        }
    with open(args.out, "w") as f:
        json.dump(obs, f, indent=1, sort_keys=True)
    print(f"wrote {args.out}")


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: only in B: {json.dumps(b[k])[:200]}")
            elif k not in b:
                out.append(f"{path}/{k}: only in A: {json.dumps(a[k])[:200]}")
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}:\n    A: {json.dumps(a)[:300]}\n    B: {json.dumps(b)[:300]}")


def diff(args):
    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    out = []
    walk(a, b, "", out)
    for line in out:
        print(line)
    print(f"{len(out)} difference(s)")
    return 1 if out else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("observe")
    for name in ("gateway", "fakevllm", "fakeoverflow", "out"):
        o.add_argument("--" + name, required=True)
    o.add_argument("--port-base", type=int, default=18100)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args()
    sys.exit(observe(args) if args.cmd == "observe" else diff(args))


if __name__ == "__main__":
    main()
