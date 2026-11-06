"""Agent-supervised inference workers under owner memory caps (FR 07 to FR 09, AT 10 in miniature).

Real difa-agent processes supervise real llama.cpp rpc-server workers against a live coordinator using
the llama.cpp adapter. Lowering a device's cap below what its worker holds must kill the worker, drain
the deployment, and keep it stopped until the owner raises the cap again. Skipped unless the agent
binary, the pinned llama.cpp build, and the model are present. Synthetic prompts only.
"""
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from difa.api.app import create_app
from difa.config import Settings
from difa.runtime.llamacpp import LlamaCppRuntime

WORK = Path.home() / ".cache" / "difa-phase1"
BIN = WORK / "llama.cpp" / "build" / "bin"
MODELS = WORK / "models"
AGENT = Path.home() / ".cache" / "difa-target" / "debug" / "difa-agent"
DIGEST = "llama.cpp-v0.9.0"
PASS = "worker cap test passphrase"
GB = 10**9

pytestmark = pytest.mark.skipif(
    not AGENT.exists() or not (BIN / "ggml-rpc-server").exists()
    or not (MODELS / "qwen2.5-0.5b-instruct-q8_0.gguf").exists(),
    reason="agent binary, llama.cpp build, or model not present")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def until(pred, timeout: float, what: str):
    end = time.time() + timeout
    while time.time() < end:
        value = pred()
        if value:
            return value
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def coordinator(tmp_path):
    port = free_port()
    rt = LlamaCppRuntime(str(BIN), str(MODELS), "Mac Studio", {}, log_dir=str(tmp_path))
    app = create_app(Settings(db_path=str(tmp_path / "difa.sqlite3")), rt)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    until(lambda: server.started, 20, "coordinator start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(20)


def test_lowering_the_cap_kills_the_worker_and_drains_the_deployment(coordinator, tmp_path):
    base = coordinator
    c = httpx.Client(base_url=base, timeout=60)
    g = c.post("/api/v1/groups", json={"name": "Cap", "owner_login": "owner", "owner_display_name": "Owner",
                                       "passphrase": PASS}).json()
    info = c.post("/api/v1/auth/login", json={"group_id": g["group_id"], "login": "owner", "passphrase": PASS}).json()
    c.headers["X-CSRF-Token"] = info["csrf_token"]
    gid = g["group_id"]

    # The coordinator's own Metal device has no agent-managed worker; keep it heartbeating from the test.
    code = c.post("/api/v1/devices/enrollment-codes").json()["code"]
    mac_caps = {"os": "macos", "arch": "aarch64", "backend_digest": DIGEST, "formats": ["gguf"],
                "allocatable_bytes": 64 * GB, "mem_bandwidth_gbps": 159, "rtt_ms": 0, "power": "ac"}
    mac = c.post("/api/v1/devices/enroll", json={"code": code, "name": "Mac Studio", "public_key": "k",
                                                 "capabilities": mac_caps,
                                                 "policy": {"memory_cap_bytes": 64 * GB}}).json()
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            httpx.post(f"{base}/api/v1/devices/{mac['device_id']}/heartbeat", json={"capabilities": mac_caps},
                       headers={"Authorization": f"Bearer {mac['device_token']}"}, timeout=10)
            stop.wait(2)
    threading.Thread(target=beat, daemon=True).start()

    agents, logs = {}, {}
    try:
        for name, profile, port, threads in [("HP EliteBook", "hp-elitebook", free_port(), 4),
                                             ("Ubuntu VM", "ubuntu-vm", free_port(), 2)]:
            code = c.post("/api/v1/devices/enrollment-codes").json()["code"]
            state = tmp_path / f"{profile}.json"
            out = subprocess.run([str(AGENT), "enroll", "--coordinator", base, "--code", code, "--name", name,
                                  "--memory-cap-gb", "1", "--state", str(state), "--simulate", profile,
                                  "--backend-digest", DIGEST], capture_output=True, text=True)
            assert out.returncode == 0, out.stderr
            logs[name] = tmp_path / f"{profile}.log"
            agents[name] = subprocess.Popen(
                [str(AGENT), "run", "--state", str(state), "--worker-bin", str(BIN / "ggml-rpc-server"),
                 "--worker-port", str(port), "--worker-threads", str(threads)],
                stdout=open(logs[name], "w"), stderr=subprocess.STDOUT)

        def ready():
            devs = c.get(f"/api/v1/groups/{gid}/devices").json()["devices"]
            return len(devs) == 3 and all(d["state"] == "ready" for d in devs) and devs
        devs = until(ready, 30, "three ready devices")
        hp = next(d for d in devs if d["name"] == "HP EliteBook")

        r = c.post(f"/api/v1/groups/{gid}/deployment", json={"model": "qwen2.5-0.5b-q8", "require_all": True})
        assert r.status_code == 200, r.text
        stages = r.json()["explanation"]["stages"]
        assert all(s["endpoint"] for s in stages if s["name"] != "Mac Studio")  # addresses came from the agents
        assert all(p["model_mib"] > 0 for p in r.json()["loaded"]["partition_map"])

        ok = c.post("/api/v1/requests", json={"model": "group-default", "visibility": "private",
                                              "max_output_tokens": 16,
                                              "messages": [{"role": "user", "content": "Say hello."}]}).json()
        until(lambda: c.get(ok["status_url"]).json()["status"] == "completed", 60, "first answer")

        # The owner lowers the HP cap below what its worker already holds.
        assert c.patch(f"/api/v1/devices/{hp['device_id']}/policy",
                       json={"memory_cap_bytes": 50_000_000}).status_code == 200
        until(lambda: "worker killed: memory cap exceeded" in logs["HP EliteBook"].read_text(), 20, "cap kill")
        until(lambda: c.get(f"/api/v1/groups/{gid}/deployment").json()["state"] == "unavailable", 20, "drain")
        r = c.post("/api/v1/requests", json={"model": "group-default", "visibility": "private",
                                             "max_output_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
        # Raising the cap back lets the worker restart and the device become ready again.
        c.patch(f"/api/v1/devices/{hp['device_id']}/policy", json={"memory_cap_bytes": 1 * GB})
        until(lambda: all(d["state"] == "ready" for d in c.get(f"/api/v1/groups/{gid}/devices").json()["devices"]),
              30, "worker restart after cap raised")
    finally:
        stop.set()
        for p in agents.values():
            p.terminate()
        for p in agents.values():
            p.wait(15)
    for name, log in logs.items():
        assert "agent stopping: worker halted" in log.read_text(), name  # no orphaned workers
