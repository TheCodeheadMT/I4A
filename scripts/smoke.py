"""Three-device smoke test against a running coordinator, using real difa-agent processes.

The agents use simulation profiles for the reference group (Mac capped at 64 GB, HP, Ubuntu VM).
It checks Test A cooperation, the Test C pilot placement, chat streaming, and the compatible API.
The fake runtime is used throughout: this proves product contracts, not distributed inference.

Usage: python scripts/smoke.py [--url http://127.0.0.1:8765] [--agent PATH]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8765")
parser.add_argument("--agent", default=str(Path.home() / ".cache" / "difa-target" / "debug" / "difa-agent"))
args = parser.parse_args()
BASE = args.url.rstrip("/")
PASS = "smoke test passphrase long enough"


class Session:
    def __init__(self):
        self.cookie = self.csrf = None

    def call(self, method, path, body=None, bearer=None, stream=False):
        headers = {"Content-Type": "application/json"}
        if self.cookie:
            headers["Cookie"], headers["X-CSRF-Token"] = self.cookie, self.csrf
        if bearer:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {bearer}"}
        req = urllib.request.Request(BASE + path, method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}"), None
        if stream:
            return resp.status, None, resp
        if not self.cookie and resp.headers.get("Set-Cookie"):
            self.cookie = resp.headers["Set-Cookie"].split(";")[0]
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else None, resp


def check(cond, label):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        sys.exit(1)


s = Session()
status, g, _ = s.call("POST", "/api/v1/groups", {"name": "Smoke Group", "owner_login": "owner",
                                                  "owner_display_name": "Owner", "passphrase": PASS})
check(status == 201, f"group created ({g.get('group_id')}, fingerprint {g.get('fingerprint')})")
status, info, _ = s.call("POST", "/api/v1/auth/login", {"group_id": g["group_id"], "login": "owner", "passphrase": PASS})
s.csrf = info["csrf_token"]
check(status == 200, "owner signed in")

tmp = Path(tempfile.mkdtemp(prefix="difa-smoke-"))
agents = []
for name, profile, cap in [("Mac Studio", "mac-studio", 64), ("HP EliteBook", "hp-elitebook", 8),
                           ("Ubuntu VM", "ubuntu-vm", 4.5)]:
    code = s.call("POST", "/api/v1/devices/enrollment-codes", {})[1]["code"]
    state = tmp / f"{profile}.json"
    out = subprocess.run([args.agent, "enroll", "--coordinator", BASE, "--code", code, "--name", name,
                          "--memory-cap-gb", str(cap), "--state", str(state), "--simulate", profile],
                         capture_output=True, text=True)
    check(out.returncode == 0, f"agent enrolled {name}: {out.stdout.strip() or out.stderr.strip()}")
    check(oct(os.stat(state).st_mode & 0o777) == "0o600", f"{name} agent state file is owner-only")
    agents.append(subprocess.Popen([args.agent, "run", "--state", str(state)], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True))
time.sleep(1.5)

try:
    devices = s.call("GET", f"/api/v1/groups/{g['group_id']}/devices")[1]["devices"]
    check(all(d["state"] == "ready" for d in devices) and len(devices) == 3, "three agents heartbeating and ready")

    status, dep, _ = s.call("POST", f"/api/v1/groups/{g['group_id']}/deployment", {"model": "pilot-test-c"})
    check(status == 200, "Test C pilot model placed")
    exp = dep["explanation"]
    for st in exp["stages"]:
        print(f"     {st['name']:<13} layers {st['layers'][0]:>2}-{st['layers'][1] - 1:<2} "
              f"weights {st['weights_bytes'] / 1e9:5.1f} GB  margin {st['margin_bytes'] / 1e9:5.2f} GB  "
              f"est {st['est_compute_ms']:6.1f} ms + rtt {st['rtt_ms']} ms")
    lb = exp["latency_budget"]
    print(f"     latency budget: {lb['per_token_ms']} ms/token (~{lb['tokens_per_s']} tok/s), "
          f"dominant stage {lb['dominant_stage']}")
    check(exp["single_device_fit"] == [] and len(exp["stages"]) == 3, exp["capacity_basis"])

    status, r, _ = s.call("POST", "/api/v1/requests", {"model": "group-default", "visibility": "private",
                                                        "max_output_tokens": 64,
                                                        "messages": [{"role": "user", "content": "Hello group"}]})
    check(status == 202, f"chat request admitted ({r['request_id']})")
    _, _, resp = s.call("GET", r["events_url"], stream=True)
    events = [json.loads(line[6:]) for line in (raw.decode().strip() for raw in resp) if line.startswith("data: ")]
    text = "".join(e["payload"]["text"] for e in events if e["type"] == "token")
    check(events[-1]["type"] == "completed", f"streamed answer: {text}")

    status, key, _ = s.call("POST", "/api/v1/credentials", {"name": "smoke app"})
    status, out, _ = s.call("POST", "/v1/chat/completions", {"model": "group-default", "max_tokens": 16,
                                                              "messages": [{"role": "user", "content": "hi"}]},
                            bearer=key["secret"])
    check(status == 200 and out["choices"][0]["message"]["content"], "OpenAI-compatible completion returned")

    agents[2].terminate()
    print("     stopped the Ubuntu VM agent; waiting for the coordinator to notice")
    deadline = time.time() + 25
    state = None
    while time.time() < deadline:
        state = s.call("GET", f"/api/v1/groups/{g['group_id']}/deployment")[1]["state"]
        if state == "unavailable":
            break
        time.sleep(1)
    check(state == "unavailable", "deployment marked unavailable after device loss")
finally:
    for a in agents:
        a.terminate()
print("smoke test complete")
