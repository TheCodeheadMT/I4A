"""Real llama.cpp adapter tests on loopback (Phase 1, AT 04 and AT 11 in miniature).

Skipped unless the pinned llama.cpp build and model are in ~/.cache/difa-phase1. RPC workers are local
CPU processes standing in for the HP and VM; prompts are synthetic.
"""
import json
import socket
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import OWNER_PASS, enroll, login
from difa.api.app import create_app
from difa.config import Settings
from difa.runtime.llamacpp import LlamaCppRuntime

WORK = Path.home() / ".cache" / "difa-phase1"
BIN = WORK / "llama.cpp" / "build" / "bin"
MODELS = WORK / "models"
DIGEST = "llama.cpp-v0.9.0"

pytestmark = pytest.mark.skipif(
    not (BIN / "llama-server").exists() or not (MODELS / "qwen2.5-0.5b-instruct-q8_0.gguf").exists(),
    reason="pinned llama.cpp build or model not present")


def _wait(port: int) -> None:
    for _ in range(150):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"rpc worker on {port} did not start")


def start_workers(ports: dict[str, tuple[int, int]]) -> dict[str, subprocess.Popen]:
    procs = {name: subprocess.Popen([str(BIN / "ggml-rpc-server"), "-H", "127.0.0.1", "-p", str(port), "-d", "CPU",
                                     "-t", str(threads)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for name, (port, threads) in ports.items()}
    for port, _ in ports.values():
        _wait(port)
    return procs


def setup_group(client):
    g = client.post("/api/v1/groups", json={"name": "LLM", "owner_login": "alice", "owner_display_name": "Alice",
                                            "passphrase": OWNER_PASS}).json()
    alice = login(client, g["group_id"], "alice", OWNER_PASS)
    devs = {
        "mac": enroll(client, alice, "Mac Studio", 110, 64, 800, 1, digest=DIGEST),
        "hp": enroll(client, alice, "HP EliteBook", 10.2, 8, 51, 5, digest=DIGEST),
        "vm": enroll(client, alice, "Ubuntu VM", 5, 4.5, 15, 60, digest=DIGEST),
    }
    return g, alice, devs


def make_client(tmp_path, ports):
    rt = LlamaCppRuntime(str(BIN), str(MODELS), "Mac Studio",
                         {name: f"127.0.0.1:{port}" for name, (port, _) in ports.items()}, log_dir=str(tmp_path))
    return TestClient(create_app(Settings(db_path=str(tmp_path / "difa.sqlite3")), rt)), rt


def events(user, admitted):
    with user.client.stream("GET", admitted["events_url"], headers=user._h()) as s:
        return [json.loads(line[6:]) for line in s.iter_lines() if line.startswith("data: ")]


def test_three_device_deployment_runs_real_inference(tmp_path):
    ports = {"HP EliteBook": (50062, 4), "Ubuntu VM": (50063, 2)}
    workers = start_workers(ports)
    client, rt = make_client(tmp_path, ports)
    try:
        with client:
            g, alice, devs = setup_group(client)
            r = alice.post(f"/api/v1/groups/{g['group_id']}/deployment",
                           json={"model": "qwen2.5-0.5b-q8", "require_all": True})
            assert r.status_code == 200, r.text
            pmap = r.json()["loaded"]["partition_map"]
            ids = {v["device_id"]: k for k, v in devs.items()}
            assert [ids[p["device_id"]] for p in pmap] == ["vm", "hp", "mac"]
            assert all(p["model_mib"] > 0 for p in pmap)  # verified from llama.cpp's own load log
            assert pmap[-1]["model_mib"] == max(p["model_mib"] for p in pmap)  # Mac holds the output head

            admitted = alice.post("/api/v1/requests", json={
                "model": "group-default", "visibility": "private", "max_output_tokens": 32, "seed": 7,
                "messages": [{"role": "user", "content": "Name three colours of the rainbow."}]}).json()
            ev = events(alice, admitted)
            assert ev[-1]["type"] == "completed", ev[-1]
            text = "".join(e["payload"]["text"] for e in ev if e["type"] == "token")
            assert text.strip() and "Simulated" not in text
            assert len(ev[1]["payload"]["partition_map"]) == 3

            key = alice.post("/api/v1/credentials", json={"name": "app"}).json()["secret"]
            out = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={
                "model": "group-default", "max_tokens": 16, "messages": [{"role": "user", "content": "Say hi."}]})
            assert out.status_code == 200 and out.json()["choices"][0]["message"]["content"].strip()
    finally:
        for p in workers.values():
            p.terminate()


def test_killing_a_worker_mid_generation_is_a_clear_failure(tmp_path):
    ports = {"HP EliteBook": (50072, 4), "Ubuntu VM": (50073, 2)}
    workers = start_workers(ports)
    client, rt = make_client(tmp_path, ports)
    try:
        with client:
            g, alice, devs = setup_group(client)
            assert alice.post(f"/api/v1/groups/{g['group_id']}/deployment",
                              json={"model": "qwen2.5-0.5b-q8", "require_all": True}).status_code == 200
            admitted = alice.post("/api/v1/requests", json={
                "model": "group-default", "visibility": "private", "max_output_tokens": 512,
                "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]}).json()
            for _ in range(200):
                if alice.get(admitted["status_url"]).json()["status"] == "running":
                    break
                time.sleep(0.02)
            workers["Ubuntu VM"].kill()
            ev = events(alice, admitted)
            assert ev[-1]["type"] in ("failed", "completed")
            if ev[-1]["type"] == "completed":
                pytest.skip("generation finished before the worker was killed")
            assert not any(e["type"] == "completed" for e in ev)
            msgs = alice.get(f"/api/v1/conversations/{admitted['branch_id']}").json()["messages"]
            assert msgs[-1]["role"] == "user" or msgs[-1]["completion_state"] == "incomplete"
    finally:
        for p in workers.values():
            p.kill()
