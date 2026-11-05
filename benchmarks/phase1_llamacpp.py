"""Phase 1 feasibility harness for llama.cpp (spec v0.3 Section 13 Phase 1, whitepaper 6.2 Stages 1 to 3).

Runs on one machine. RPC workers are CPU-only llama.cpp rpc-server processes on 127.0.0.1, used as
stand-ins for remote CPU contributors. Loopback has near-zero network latency and the stand-ins run on
M2 Ultra cores, so these results establish partition execution and per-hop software overhead, NOT the
speed of the real HP laptop, the remote VM, or an internet path.

Security: the llama.cpp RPC server is documented upstream as fragile and insecure. Everything binds to
127.0.0.1 and every prompt is synthetic.

Acceptance rule fixed before results (whitepaper 6.2 Stage 2):
  * every request completes with non-empty output, and
  * with greedy decoding, at least 90 percent of prompts produce the same first 32 tokens as the
    single-device Metal baseline. Divergence is reported, not hidden; CPU and Metal kernels may differ
    numerically.

Usage: python benchmarks/phase1_llamacpp.py [--requests 100] [--quick]
"""
import argparse
import hashlib
import json
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path.home() / ".cache" / "difa-phase1"
BIN = WORK / "llama.cpp" / "build" / "bin"
MODEL = WORK / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
MODEL_SHA256 = "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
SERVER_PORT = 8790
PROMPT_TOKENS, OUTPUT_TOKENS, COMPARE_TOKENS, MATCH_THRESHOLD = 256, 128, 32, 0.90

WORKERS = {  # name -> (port, threads); CPU-only stand-ins for remote contributors
    "cpu4": (50052, 4),
    "cpu2": (50053, 2),
}

# The cooperation split mirrors the pilot placement shape (Mac 57, HP 5, VM 2 of 64 layers).
TOPOLOGIES = [
    {"name": "mac-metal", "label": "Single device, Metal (baseline)", "workers": [], "dev": "MTL0", "ts": None},
    {"name": "mac-cpu4", "label": "Single device, CPU 4 threads (not an HP measurement)", "workers": [],
     "dev": "none", "ts": None, "threads": 4},
    {"name": "coop-2", "label": "Metal + 1 RPC CPU worker", "workers": ["cpu4"], "dev": "MTL0,RPC0", "ts": "0.8,0.2"},
    # A 0.89/0.08/0.03 split rounds the third worker to zero of 24 layers (observed 2026-11-05), so the
    # cooperation case uses shares that give every worker at least one layer; the harness verifies it.
    {"name": "coop-3", "label": "Metal + 2 RPC CPU workers (Test A shape)", "workers": ["cpu4", "cpu2"],
     "dev": "MTL0,RPC0,RPC1", "ts": "0.80,0.12,0.08"},
]

TOPICS = ["river ecology", "bread baking", "bicycle repair", "solar panels", "chess openings", "tide pools",
          "library catalogues", "volcanoes", "bird migration", "paper making", "rail signalling", "honey bees",
          "glass blowing", "sea kayaking", "desert plants", "clock repair", "map projections", "cheese aging",
          "kite design", "coral reefs"]


def wait_port(port: int, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"port {port} did not open")


def http(method: str, path: str, body: dict | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{SERVER_PORT}{path}", method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bench(topo: dict, reps: int) -> dict:
    args = [str(BIN / "llama-bench"), "-m", str(MODEL), "-p", str(PROMPT_TOKENS), "-n", str(OUTPUT_TOKENS),
            "-r", str(reps), "-o", "json"]
    if topo["workers"]:
        args += ["-rpc", ",".join(f"127.0.0.1:{WORKERS[w][0]}" for w in topo["workers"]),
                 "-dev", topo["dev"].replace(",", "/"), "-ts", topo["ts"].replace(",", "/")]
    elif topo["dev"] == "none":
        args += ["-ngl", "0", "-t", str(topo["threads"])]
    out = subprocess.run(args, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        return {"error": out.stderr[-2000:]}
    rows = json.loads(out.stdout)
    res = {}
    for r in rows:
        key = "prompt_tps" if r["n_prompt"] else "decode_tps"
        res[key] = round(r["avg_ts"], 1)
        res[key + "_sd"] = round(r["stddev_ts"], 1)
    return res


def make_prompts(n: int) -> list[str]:
    prompts = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        body = (f"Synthetic test document {i}. This passage describes {topic} for a benchmark. " * 40)
        prompts.append(body)
    return prompts


def trim_to_tokens(prompt: str, tokens: int) -> str:
    toks = http("POST", "/tokenize", {"content": prompt})["tokens"][:tokens]
    return http("POST", "/detokenize", {"tokens": toks})["content"]


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return round(int(out or 0) / 1024, 1)


def serve(topo: dict, prompts: list[str], log_path: Path, pids: dict) -> dict:
    args = [str(BIN / "llama-server"), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(SERVER_PORT),
            "-c", "4096", "-np", "1", "--seed", "42", "--no-webui", "-lv", "4"]
    if topo["workers"]:
        args += ["--rpc", ",".join(f"127.0.0.1:{WORKERS[w][0]}" for w in topo["workers"]),
                 "-dev", topo["dev"], "-ts", topo["ts"], "-ngl", "99"]
    elif topo["dev"] == "none":
        args += ["-ngl", "0", "-t", str(topo["threads"])]
    else:
        args += ["-dev", topo["dev"], "-ngl", "99"]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_port(SERVER_PORT, 120)
            for _ in range(600):
                try:
                    if http("GET", "/health").get("status") == "ok":
                        break
                except Exception:
                    pass
                time.sleep(0.2)
            worker_rss = {w: rss_mb(pids[w]) for w in topo["workers"]}  # weights resident on each worker
            fixed = [trim_to_tokens(p, PROMPT_TOKENS) for p in prompts]
            http("POST", "/completion", {"prompt": fixed[0], "n_predict": 8, "temperature": 0})  # warm-up
            results = []
            for p in fixed:
                t0 = time.perf_counter()
                r = http("POST", "/completion", {"prompt": p, "n_predict": OUTPUT_TOKENS, "temperature": 0,
                                                 "top_k": 1, "seed": 42, "cache_prompt": False,
                                                 "return_tokens": True})
                wall = time.perf_counter() - t0
                t = r["timings"]
                results.append({"tokens": r.get("tokens", []), "text_len": len(r["content"]),
                                "prompt_ms": t["prompt_ms"], "predicted_tps": t["predicted_per_second"],
                                "first_token_ms": t["prompt_ms"] + t["predicted_ms"] / max(1, t["predicted_n"]),
                                "wall_s": wall, "n": t["predicted_n"]})
            return {"results": results, "worker_rss_mb": worker_rss}
        finally:
            proc.terminate()
            proc.wait(timeout=30)


def partition_evidence(log_path: Path) -> list[str]:
    lines = log_path.read_text(errors="replace").splitlines()
    keep = re.compile(r"(load_tensors: offloaded|model buffer size|KV buffer size|compute buffer size)", re.I)
    seen, out = set(), []
    for ln in lines:
        text = ln.strip().split(" I ", 1)[-1]
        if keep.search(ln) and " 0.00 MiB" not in ln and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def model_buffers(evidence: list[str]) -> dict[str, float]:
    """MiB of model weights per device, parsed from llama.cpp load_tensors lines.

    llama.cpp v0.9.0 names every RPC buffer "RPC0[host:port]", so RPC devices are keyed by endpoint.
    """
    out: dict[str, float] = {}
    for ln in evidence:
        m = re.search(r"load_tensors:\s+(\S+?)(?:\[([^\]]*)\])?(?:_Mapped)? model buffer size =\s+([\d.]+) MiB", ln)
        if m:
            key = f"rpc:{m.group(2)}" if m.group(2) else m.group(1)
            out[key] = round(out.get(key, 0) + float(m.group(3)), 2)
    return out


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=100, help="warm requests per topology (spec minimum 100)")
    ap.add_argument("--quick", action="store_true", help="10 requests and 1 bench repetition, for smoke runs")
    a = ap.parse_args()
    n, reps = (10, 1) if a.quick else (a.requests, 3)

    if sha256(MODEL) != MODEL_SHA256:
        sys.exit("model hash mismatch")
    commit = subprocess.run(["git", "-C", str(WORK / "llama.cpp"), "log", "-1", "--format=%H"],
                            capture_output=True, text=True).stdout.strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "benchmarks" / "results"
    logs = WORK / "logs" / stamp
    logs.mkdir(parents=True, exist_ok=True)

    workers = {}
    for name, (port, threads) in WORKERS.items():
        log = open(logs / f"rpc-{name}.log", "w")
        workers[name] = subprocess.Popen([str(BIN / "ggml-rpc-server"), "-H", "127.0.0.1", "-p", str(port),
                                          "-d", "CPU", "-t", str(threads)], stdout=log, stderr=subprocess.STDOUT)
    report = {
        "harness": "phase1_llamacpp/1", "utc": stamp, "synthetic_prompts": True,
        "environment": {"machine": platform.machine(), "os": platform.platform(),
                        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                               text=True).stdout.strip()},
        "llama_cpp": {"tag": "v0.9.0", "commit": commit},
        "model": {"file": MODEL.name, "sha256": MODEL_SHA256, "source": "huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                  "license": "apache-2.0"},
        "workload": {"prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS, "requests": n,
                     "decoding": "greedy, seed 42, prompt cache disabled"},
        "acceptance": {"compare_tokens": COMPARE_TOKENS, "match_threshold": MATCH_THRESHOLD},
        "limitations": ["loopback network only", "RPC workers are CPU processes on the same M2 Ultra, not the HP or VM",
                        "single small model; not a capacity (Test B or Test C) result"],
        "topologies": [],
    }
    try:
        for name, (port, _) in WORKERS.items():
            wait_port(port)
        prompts = make_prompts(n)
        baseline_tokens = None
        for topo in TOPOLOGIES:
            print(f"== {topo['name']}: {topo['label']}", flush=True)
            b = bench(topo, reps)
            log_path = logs / f"server-{topo['name']}.log"
            idle = {w: rss_mb(workers[w].pid) for w in topo["workers"]}
            s = serve(topo, prompts, log_path, {w: p.pid for w, p in workers.items()})
            res = s["results"]
            toks = [r["tokens"][:COMPARE_TOKENS] for r in res]
            if baseline_tokens is None:
                baseline_tokens = toks
            matches = sum(1 for x, y in zip(toks, baseline_tokens) if x == y)
            entry = {
                "name": topo["name"], "label": topo["label"], "devices": topo["dev"], "tensor_split": topo["ts"],
                "bench": b,
                "requests": len(res),
                "all_nonempty": all(r["n"] > 0 and r["text_len"] > 0 for r in res),
                "first_token_ms_p50": pct([r["first_token_ms"] for r in res], 0.5),
                "first_token_ms_p95": pct([r["first_token_ms"] for r in res], 0.95),
                "decode_tps_median": round(statistics.median(r["predicted_tps"] for r in res), 1),
                "wall_s_p95": round(pct([r["wall_s"] * 1000 for r in res], 0.95) / 1000, 3),
                "greedy_match_rate": round(matches / len(res), 3),
                "partition_evidence": partition_evidence(log_path),
                "worker_rss_mb": {w: {"idle": idle[w], "loaded": s["worker_rss_mb"][w]} for w in topo["workers"]},
            }
            entry["model_buffers_mib"] = model_buffers(entry["partition_evidence"])
            # Every RPC worker named in the topology must hold model weights; registration alone is not evidence.
            entry["all_workers_hold_weights"] = all(
                entry["model_buffers_mib"].get(f"rpc:127.0.0.1:{WORKERS[w][0]}", 0) > 0 for w in topo["workers"])
            entry["accepted"] = (entry["all_nonempty"] and entry["greedy_match_rate"] >= MATCH_THRESHOLD
                                 and entry["all_workers_hold_weights"])
            report["topologies"].append(entry)
            print(json.dumps({k: entry[k] for k in ("bench", "model_buffers_mib", "first_token_ms_p95",
                                                    "decode_tps_median", "greedy_match_rate", "accepted")}), flush=True)
    finally:
        for p in workers.values():
            p.terminate()

    base = report["topologies"][0]["bench"].get("decode_tps")
    for t in report["topologies"]:
        tps = t["bench"].get("decode_tps")
        if base and tps:
            t["added_ms_per_token_vs_baseline"] = round(1000 / tps - 1000 / base, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"phase1-llamacpp-{stamp}.json"
    jpath.write_text(json.dumps(report, indent=2) + "\n")
    md = [f"# Phase 1 llama.cpp loopback results ({stamp})", "",
          f"llama.cpp {report['llama_cpp']['tag']} ({commit[:10]}), model {MODEL.name}, "
          f"{n} warm requests per topology, {PROMPT_TOKENS}-token prompts, {OUTPUT_TOKENS} output tokens, greedy.",
          "", "Limitations: " + "; ".join(report["limitations"]) + ".", "",
          "| Topology | Weights per device (MiB) | Prompt t/s | Decode t/s | Added ms/token | First token p95 ms | Greedy match | Accepted |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for t in report["topologies"]:
        b = t["bench"]
        weights = ", ".join(f"{k} {v:g}" for k, v in t["model_buffers_mib"].items())
        md.append(f"| {t['label']} | {weights} | {b.get('prompt_tps', 'err')} | {b.get('decode_tps', 'err')} | "
                  f"{t.get('added_ms_per_token_vs_baseline', '')} | {t['first_token_ms_p95']} | "
                  f"{t['greedy_match_rate']:.0%} | {'yes' if t['accepted'] else 'NO'} |")
    (out_dir / f"phase1-llamacpp-{stamp}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {jpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
