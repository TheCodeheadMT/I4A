"""Latency-model calibration, hold-out validation, and simulated network paths (spec 7.2, whitepaper 4.3).

Model under test, per generated token:
    t = sum_i W_i / BW_b(i)  +  O_first + O_extra * (remote_stages - 1)  +  k * sum_remote RTT_i
where W_i is the weight bytes llama.cpp actually placed on stage i, BW_b is the effective decode
bandwidth of that stage's backend (from single-device runs), O_* are fixed RPC overheads, and k is the
number of dependent round trips per token per remote stage.

Constants are fitted on calibration topologies only and then checked on held-out topologies against the
spec's 30 percent agreement target. Everything runs on one Mac: RPC workers are local CPU processes and
network paths come from benchmarks/netem_relay.py, so this validates the model's shape, not real-world
speeds. Prompts are synthetic. Usage: python benchmarks/phase1_latency.py
"""
import asyncio
import json
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from netem_relay import DelayRelay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = Path.home() / ".cache" / "difa-phase1"
BIN = WORK / "llama.cpp" / "build" / "bin"
MODEL = WORK / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
PORT = 8795
MIB = 1 << 20
TARGET = 0.30
WORKERS = {"w1": (50082, 4), "w2": (50083, 4), "w3": (50084, 2)}  # name -> (port, threads)
PROMPT = "Synthetic latency benchmark text about tide pools and coral reefs. " * 30


class Relays:
    """Relays run on an asyncio loop in a background thread so benchmarks stay synchronous."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.active: dict[int, DelayRelay] = {}

    def start(self, listen: int, target: int, rtt_ms: float, mbps: float | None = None) -> None:
        relay = DelayRelay(listen, target, rtt_ms, mbps)
        asyncio.run_coroutine_threadsafe(relay.start(), self.loop).result(10)
        self.active[listen] = relay

    def stop_all(self) -> None:
        for relay in self.active.values():
            asyncio.run_coroutine_threadsafe(relay.stop(), self.loop).result(10)
        self.active.clear()


def wait_port(port: int, timeout: float = 120) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} closed")


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def buffers(log: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in re.finditer(r"load_tensors:\s+(\S+?)(?:\[([^\]]*)\])?(?:_Mapped)? model buffer size =\s+([\d.]+) MiB", log):
        key = m.group(2) or m.group(1)
        out[key] = round(out.get(key, 0) + float(m.group(3)), 2)
    return out


def measure(name: str, dev: str, ts: str | None, endpoints: list[str], threads: int | None, requests: int,
            n_predict: int, log_dir: Path) -> dict:
    """Load a topology in llama-server, then time greedy decoding. Returns buffers and timings."""
    args = [str(BIN / "llama-server"), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(PORT), "-c", "2048",
            "-np", "1", "--no-webui", "-lv", "4", "-fit", "off"]
    if endpoints:
        args += ["--rpc", ",".join(endpoints)]
    if dev == "none":
        args += ["-ngl", "0", "-t", str(threads)]
    else:
        args += ["-ngl", "99", "-dev", dev] + (["-ts", ts] if ts else [])
    log_path = log_dir / f"{name}.log"
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_port(PORT, 900)
        while True:
            try:
                if call("/health").get("status") == "ok":
                    break
            except Exception:
                time.sleep(0.2)
        load_s = time.perf_counter() - t0
        call("/completion", {"prompt": PROMPT, "n_predict": 4, "temperature": 0})  # warm-up
        decode_ms, prompt_ms = [], []
        for _ in range(requests):
            r = call("/completion", {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                                     "cache_prompt": False, "ignore_eos": True})
            t = r["timings"]
            decode_ms.append(t["predicted_ms"] / max(1, t["predicted_n"]))
            prompt_ms.append(t["prompt_ms"])
        return {"name": name, "load_s": round(load_s, 2), "decode_ms": round(statistics.median(decode_ms), 3),
                "prompt_ms": round(statistics.median(prompt_ms), 1), "buffers_mib": buffers(log_path.read_text())}
    finally:
        proc.terminate()
        proc.wait(30)


def ep(worker: str, via: int | None = None) -> str:
    return f"127.0.0.1:{via or WORKERS[worker][0]}"


def fit_slope(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = WORK / "logs" / f"latency-{stamp}"
    logs.mkdir(parents=True, exist_ok=True)
    procs = [subprocess.Popen([str(BIN / "ggml-rpc-server"), "-H", "127.0.0.1", "-p", str(port), "-d", "CPU",
                               "-t", str(th)], stdout=open(logs / f"rpc-{w}.log", "w"), stderr=subprocess.STDOUT)
             for w, (port, th) in WORKERS.items()]
    relays = Relays()
    runs: dict[str, dict] = {}
    try:
        for port, _ in WORKERS.values():
            wait_port(port)

        def run(name, dev, ts=None, endpoints=(), threads=None, requests=5, n_predict=64):
            print(f"== {name}", flush=True)
            runs[name] = measure(name, dev, ts, list(endpoints), threads, requests, n_predict, logs)
            print(json.dumps({k: runs[name][k] for k in ("load_s", "decode_ms", "prompt_ms", "buffers_mib")}), flush=True)

        # Single-device runs give each backend's effective decode bandwidth.
        run("single-metal", "MTL0")
        run("single-cpu4", "none", threads=4)
        run("single-cpu2", "none", threads=2)
        # Calibration topologies (fit O_first and O_extra).
        run("cal-2", "MTL0,RPC0", "0.8,0.2", [ep("w1")])
        run("cal-3", "MTL0,RPC0,RPC1", "0.80,0.12,0.08", [ep("w1"), ep("w3")])
        # Held-out topologies (never used for fitting).
        run("val-2-even", "MTL0,RPC0", "0.5,0.5", [ep("w1")])
        run("val-3", "MTL0,RPC0,RPC1", "0.6,0.2,0.2", [ep("w1"), ep("w2")])
        run("val-4", "MTL0,RPC0,RPC1,RPC2", "0.55,0.15,0.15,0.15", [ep("w1"), ep("w2"), ep("w3")])
        # Simulated paths: fit k from the 2-stage topology across round-trip times.
        for rtt in (0, 20, 80, 150):
            relays.start(60082, WORKERS["w1"][0], rtt)
            run(f"rtt-{rtt}", "MTL0,RPC0", "0.8,0.2", [ep("w1", 60082)], requests=3 if rtt else 5,
                n_predict=32 if rtt >= 80 else 64)
            relays.stop_all()
        # Held-out: two remote stages, each behind 80 ms.
        relays.start(60082, WORKERS["w1"][0], 80)
        relays.start(60084, WORKERS["w3"][0], 80)
        run("val-3-rtt80", "MTL0,RPC0,RPC1", "0.80,0.12,0.08", [ep("w1", 60082), ep("w3", 60084)], requests=3,
            n_predict=32)
        relays.stop_all()
        # Bandwidth caps change model loading, which streams weights over the RPC link.
        for mbps in (100, 50, 25):
            relays.start(60082, WORKERS["w1"][0], 20, mbps)
            run(f"bw-{mbps}", "MTL0,RPC0", "0.8,0.2", [ep("w1", 60082)], requests=3, n_predict=32)
            relays.stop_all()
    finally:
        relays.stop_all()
        for p in procs:
            p.terminate()

    # --- Fit -----------------------------------------------------------------------------------------
    metal_mib = sum(runs["single-metal"]["buffers_mib"].get(k, 0) for k in ("MTL0",))
    cpu_mib = sum(v for k, v in runs["single-cpu4"]["buffers_mib"].items() if k != "CPU")
    bw = {  # MiB per ms of decode
        "MTL0": metal_mib / runs["single-metal"]["decode_ms"],
        "w1": cpu_mib / runs["single-cpu4"]["decode_ms"], "w2": cpu_mib / runs["single-cpu4"]["decode_ms"],
        "w3": cpu_mib / runs["single-cpu2"]["decode_ms"],
    }
    port_to_worker = {str(p): w for w, (p, _) in WORKERS.items()}
    relay_to_worker = {"60082": "w1", "60084": "w3"}

    def compute_ms(run_: dict) -> tuple[float, int]:
        total, remote = 0.0, 0
        for dev, mib in run_["buffers_mib"].items():
            if dev == "CPU":
                continue  # host-mapped token embeddings: a lookup, not per-token weight streaming
            if dev == "MTL0":
                total += mib / bw["MTL0"]
            else:
                port = dev.split(":")[-1]
                w = port_to_worker.get(port) or relay_to_worker[port]
                total += mib / bw[w]
                remote += 1
        return total, remote

    c2, r2 = compute_ms(runs["cal-2"])
    c3, r3 = compute_ms(runs["cal-3"])
    o_first = runs["cal-2"]["decode_ms"] - c2
    o_extra = (runs["cal-3"]["decode_ms"] - c3) - o_first
    rtts = [0, 20, 80, 150]
    k = fit_slope(rtts, [runs[f"rtt-{r}"]["decode_ms"] for r in rtts])
    relay_floor = runs["rtt-0"]["decode_ms"] - runs["cal-2"]["decode_ms"]

    def predict(run_: dict, rtt_per_remote: float = 0.0, relay: bool = False) -> float:
        comp, remote = compute_ms(run_)
        return (comp + o_first + o_extra * (remote - 1) + k * rtt_per_remote * remote
                + (relay_floor * remote if relay else 0))

    checks = []
    for name, rtt, relay in [("val-2-even", 0, False), ("val-3", 0, False), ("val-4", 0, False),
                             ("val-3-rtt80", 80, True)]:
        pred, meas = predict(runs[name], rtt, relay), runs[name]["decode_ms"]
        err = (pred - meas) / meas
        checks.append({"topology": name, "predicted_ms": round(pred, 2), "measured_ms": meas,
                       "error_pct": round(err * 100, 1), "within_target": abs(err) <= TARGET})
    naive = []
    for name in ("val-2-even", "val-3", "val-4"):
        comp, _ = compute_ms(runs[name])
        meas = runs[name]["decode_ms"]
        naive.append({"topology": name, "bandwidth_only_ms": round(comp, 2), "measured_ms": meas,
                      "error_pct": round((comp - meas) / meas * 100, 1)})

    report = {
        "harness": "phase1_latency/1", "utc": stamp, "synthetic_prompts": True,
        "constants": {"bw_mib_per_ms": {k_: round(v, 2) for k_, v in bw.items()},
                      "bw_gb_per_s": {k_: round(v * MIB / 1e6, 1) for k_, v in bw.items()},
                      "o_first_ms": round(o_first, 3), "o_extra_ms": round(o_extra, 3),
                      "round_trips_per_token_per_remote_stage": round(k, 3), "relay_floor_ms": round(relay_floor, 3)},
        "validation": checks, "bandwidth_only_model": naive,
        "rtt_sweep": [{"rtt_ms": r, "decode_ms": runs[f"rtt-{r}"]["decode_ms"], "prompt_ms": runs[f"rtt-{r}"]["prompt_ms"],
                       "load_s": runs[f"rtt-{r}"]["load_s"]} for r in rtts],
        "bandwidth_caps": [{"mbps": m, "rtt_ms": 20, "load_s": runs[f"bw-{m}"]["load_s"],
                            "decode_ms": runs[f"bw-{m}"]["decode_ms"], "prompt_ms": runs[f"bw-{m}"]["prompt_ms"],
                            "remote_weights_mib": sum(v for d, v in runs[f"bw-{m}"]["buffers_mib"].items()
                                                      if d not in ("CPU", "MTL0"))} for m in (100, 50, 25)],
        "runs": runs,
        "limitations": ["all workers are local CPU processes", "network paths are simulated by a userspace relay",
                        "one small model (Qwen2.5-0.5B Q8_0)"],
    }
    out = ROOT / "benchmarks" / "results" / f"phase1-latency-{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"constants": report["constants"], "validation": checks}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
