import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DIFA_ARGON2_FAST", "1")

from difa.api.app import create_app  # noqa: E402
from difa.config import Settings  # noqa: E402
from difa.runtime.fake import FakeRuntime  # noqa: E402

GB = 10**9
OWNER_PASS = "correct horse battery staple"
MEMBER_PASS = "another long passphrase here"


class User:
    """One signed-in member. Sessions are passed explicitly so several users share one client."""

    def __init__(self, client: TestClient, info: dict, sid: str):
        self.client, self.sid = client, sid
        self.member_id, self.group_id, self.csrf = info["member_id"], info["group_id"], info["csrf_token"]

    def _h(self, extra=None):
        return {"Cookie": f"difa_session={self.sid}", "X-CSRF-Token": self.csrf, **(extra or {})}

    def get(self, url, **kw):
        return self.client.get(url, headers=self._h(kw.pop("headers", None)), **kw)

    def post(self, url, json=None, **kw):
        return self.client.post(url, json=json, headers=self._h(kw.pop("headers", None)), **kw)

    def patch(self, url, json=None):
        return self.client.patch(url, json=json, headers=self._h())

    def delete(self, url):
        return self.client.delete(url, headers=self._h())


def login(client: TestClient, group_id: str, login_name: str, passphrase: str) -> User:
    r = client.post("/api/v1/auth/login", json={"group_id": group_id, "login": login_name, "passphrase": passphrase})
    assert r.status_code == 200, r.text
    sid = r.cookies.get("difa_session")
    client.cookies.clear()
    return User(client, r.json(), sid)


@pytest.fixture
def runtime():
    return FakeRuntime(token_delay_s=0.002)


@pytest.fixture
def app(tmp_path, runtime):
    settings = Settings(db_path=str(tmp_path / "difa.sqlite3"))
    return create_app(settings, runtime)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def group(client):
    """Owner alice plus member bob in one group."""
    r = client.post("/api/v1/groups", json={"name": "Test Group", "owner_login": "alice",
                                            "owner_display_name": "Alice", "passphrase": OWNER_PASS})
    assert r.status_code == 201, r.text
    g = r.json()
    alice = login(client, g["group_id"], "alice", OWNER_PASS)
    code = alice.post(f"/api/v1/groups/{g['group_id']}/invites", json={}).json()["code"]
    r = client.post("/api/v1/invitations/redeem", json={"code": code, "login": "bob", "display_name": "Bob",
                                                        "passphrase": MEMBER_PASS})
    assert r.status_code == 201, r.text
    bob = login(client, g["group_id"], "bob", MEMBER_PASS)
    return {"group": g, "alice": alice, "bob": bob}


def enroll(client: TestClient, user: User, name: str, allocatable_gb: float, cap_gb: float, bw: float, rtt: float,
           power: str = "ac", allow_battery: bool = False, paused: bool = False,
           digest: str = "fake-runtime-0.1") -> dict:
    code = user.post("/api/v1/devices/enrollment-codes").json()["code"]
    caps = {"os": "test", "arch": "x86_64", "backend_digest": digest, "formats": ["gguf"],
            "allocatable_bytes": int(allocatable_gb * GB), "mem_bandwidth_gbps": bw, "rtt_ms": rtt, "power": power}
    r = client.post("/api/v1/devices/enroll", json={
        "code": code, "name": name, "public_key": "test-key", "capabilities": caps,
        "policy": {"memory_cap_bytes": int(cap_gb * GB), "allow_battery": allow_battery, "paused": paused}})
    assert r.status_code == 201, r.text
    dev = r.json()
    hb = client.post(f"/api/v1/devices/{dev['device_id']}/heartbeat", json={"capabilities": caps, "paused": paused},
                     headers={"Authorization": f"Bearer {dev['device_token']}"})
    assert hb.status_code == 200, hb.text
    dev["capabilities"] = caps
    return dev


@pytest.fixture
def reference_devices(client, group):
    """The v0.3 pilot: Mac capped at 64 GB by its owner, HP and VM at measured budgets (spec 3.2)."""
    alice, bob = group["alice"], group["bob"]
    return {
        "mac": enroll(client, alice, "Mac Studio", 110, 64, 800, 1),
        "hp": enroll(client, bob, "HP EliteBook", 10.2, 8, 51, 5),
        "vm": enroll(client, alice, "Ubuntu VM", 5, 4.5, 15, 60),
    }


@pytest.fixture
def deployed(client, group, reference_devices):
    r = group["alice"].post(f"/api/v1/groups/{group['group']['group_id']}/deployment",
                            json={"model": "fixture-small", "require_all": True})
    assert r.status_code == 200, r.text
    return r.json()
