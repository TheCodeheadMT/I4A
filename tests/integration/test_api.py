"""End-to-end coordinator tests through HTTP with the fake runtime (AT 01 to AT 12 where applicable)."""
import json
import threading
import time

from conftest import GB, MEMBER_PASS, enroll, login

CANARY = "CANARY-7f3a9-private-fact"


def sse(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]


def run(user, body):
    r = user.post("/api/v1/requests", json=body)
    assert r.status_code == 202, r.text
    with user.client.stream("GET", r.json()["events_url"], headers=user._h()) as s:
        events = sse(s)
    return r.json(), events


def ask(text, visibility="private", **kw):
    return {"model": "group-default", "messages": [{"role": "user", "content": text}], "visibility": visibility,
            "max_output_tokens": 64, **kw}


# --- Groups, invitations, sessions ------------------------------------------------------------------
def test_invitation_single_use_and_fingerprint(client, group):
    alice, gid = group["alice"], group["group"]["group_id"]
    code = alice.post(f"/api/v1/groups/{gid}/invites", json={}).json()["code"]
    info = client.post("/api/v1/invitations/inspect", json={"code": code}).json()
    assert info["group_fingerprint"] == group["group"]["fingerprint"] and "administrators" in info["privacy_notice"]
    body = {"code": code, "login": "carol", "display_name": "Carol", "passphrase": MEMBER_PASS}
    assert client.post("/api/v1/invitations/redeem", json=body).status_code == 201
    replay = client.post("/api/v1/invitations/redeem", json={**body, "login": "dave"})
    assert replay.status_code == 404 and replay.json()["error"]["code"] == "INVITATION_INVALID"
    wrong = client.post("/api/v1/invitations/inspect", json={"code": code.split(".")[0] + ".guess"})
    assert wrong.status_code == 404


def test_simultaneous_redemption_succeeds_once(client, group):
    alice, gid = group["alice"], group["group"]["group_id"]
    code = alice.post(f"/api/v1/groups/{gid}/invites", json={}).json()["code"]
    results = []

    def go(i):
        r = client.post("/api/v1/invitations/redeem",
                        json={"code": code, "login": f"racer{i}", "display_name": "R", "passphrase": MEMBER_PASS})
        results.append(r.status_code)
    threads = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == [201] + [404] * 5


def test_revoked_invitation_and_member_permissions(client, group):
    alice, bob, gid = group["alice"], group["bob"], group["group"]["group_id"]
    assert bob.post(f"/api/v1/groups/{gid}/invites", json={}).status_code == 403
    inv = alice.post(f"/api/v1/groups/{gid}/invites", json={}).json()
    assert alice.delete(f"/api/v1/groups/{gid}/invites/{inv['invitation_id']}").status_code == 204
    assert client.post("/api/v1/invitations/inspect", json={"code": inv["code"]}).status_code == 404


def test_csrf_required_for_cookie_writes(client, group):
    bob = group["bob"]
    r = client.post("/api/v1/conversations", json={"visibility": "private"},
                    headers={"Cookie": f"difa_session={bob.sid}"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "CSRF"


def test_bad_login_is_generic(client, group):
    r = client.post("/api/v1/auth/login", json={"group_id": group["group"]["group_id"], "login": "nobody",
                                                "passphrase": "x"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_LOGIN"


def test_recovery_rotates_and_revokes(client, group):
    gid = group["group"]["group_id"]
    code = group["alice"].post(f"/api/v1/groups/{gid}/invites", json={}).json()["code"]
    rec = client.post("/api/v1/invitations/redeem", json={"code": code, "login": "erin", "display_name": "Erin",
                                                          "passphrase": MEMBER_PASS}).json()["recovery_code"]
    erin = login(client, gid, "erin", MEMBER_PASS)
    r = client.post("/api/v1/auth/recover", json={"group_id": gid, "login": "erin", "recovery_code": rec,
                                                  "new_passphrase": "a brand new passphrase"})
    assert r.status_code == 200 and r.json()["recovery_code"] != rec
    assert erin.get("/api/v1/auth/session").status_code == 401
    login(client, gid, "erin", "a brand new passphrase")


# --- Chat, streaming, privacy ---------------------------------------------------------------------
def test_streamed_request_lifecycle_and_partition_evidence(client, group, deployed):
    alice = group["alice"]
    admitted, events = run(alice, ask("hello there"))
    types = [e["type"] for e in events]
    assert types[0] == "queued" and types[1] == "started" and types[-1] == "completed"
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert len(events[1]["payload"]["partition_map"]) == 3  # all three devices hold layers (Test A topology)
    status = alice.get(admitted["status_url"]).json()
    assert status["status"] == "completed" and status["usage"]["completion_tokens"] > 0
    conv = alice.get(f"/api/v1/conversations/{admitted['branch_id']}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    # Second turn carries the history.
    _, events2 = run(alice, ask("and again", branch_id=admitted["branch_id"]))
    text = "".join(e["payload"]["text"] for e in events2 if e["type"] == "token")
    assert "3 context messages" in text


def test_private_canary_never_reaches_another_member(client, group, deployed):
    alice, bob, gid = group["alice"], group["bob"], group["group"]["group_id"]
    admitted, _ = run(bob, ask(f"my secret is {CANARY}"))
    rid, bid = admitted["request_id"], admitted["branch_id"]
    surfaces = [
        alice.get("/api/v1/conversations"),
        alice.get(f"/api/v1/conversations/{bid}"),
        alice.get(f"/api/v1/requests/{rid}"),
        alice.get(f"/api/v1/requests/{rid}/events"),
        alice.get(f"/api/v1/groups/{gid}/queue"),
        alice.get(f"/api/v1/groups/{gid}/deployment"),
        alice.get(f"/api/v1/groups/{gid}/members"),
    ]
    for r in surfaces:
        assert CANARY not in r.text
    assert surfaces[1].status_code == 404 and surfaces[2].status_code == 404 and surfaces[3].status_code == 404
    # Administrators may cancel by id without reading text, but get nothing back about content.
    assert CANARY not in alice.post(f"/api/v1/requests/{rid}/cancel").text
    store = client.app.state.store
    assert all(CANARY not in (row["result"] or "") for row in store.all("SELECT result FROM audit"))
    assert CANARY in bob.get(f"/api/v1/conversations/{bid}").text


def test_shared_branch_visible_and_single_writer(client, group, deployed):
    alice, bob = group["alice"], group["bob"]
    conv = alice.post("/api/v1/conversations", json={"visibility": "group", "title": "Team"}).json()
    run(alice, ask("shared question", "group", branch_id=conv["branch_id"]))
    assert "shared question" in bob.get(f"/api/v1/conversations/{conv['branch_id']}").text
    r = bob.post("/api/v1/requests", json=ask("can I write?", "group", branch_id=conv["branch_id"]))
    assert r.status_code == 403 and r.json()["error"]["code"] == "SINGLE_WRITER"
    conflict = alice.post("/api/v1/requests", json=ask("mismatch", "private", branch_id=conv["branch_id"]))
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "VISIBILITY_CONFLICT"


def test_private_to_shared_fork_excludes_hidden_context(client, group, deployed):
    alice, bob = group["alice"], group["bob"]
    admitted, _ = run(alice, ask(f"private detail {CANARY}"))
    bid = admitted["branch_id"]
    msgs = alice.get(f"/api/v1/conversations/{bid}").json()["messages"]
    blocked = alice.post(f"/api/v1/conversations/{bid}/fork", json={"visibility": "group", "include_all": True})
    assert blocked.status_code == 409
    empty = alice.post(f"/api/v1/conversations/{bid}/fork", json={"visibility": "group"}).json()
    assert empty["copied_message_ids"] == []
    assert CANARY not in bob.get(f"/api/v1/conversations/{empty['branch_id']}").text
    # Publishing the assistant reply only, after preview, copies exactly that message.
    picked = alice.post(f"/api/v1/conversations/{bid}/fork", json={
        "visibility": "group", "include_message_ids": [msgs[1]["message_id"]], "preview_confirmed": True}).json()
    shared = bob.get(f"/api/v1/conversations/{picked['branch_id']}").json()
    assert len(shared["messages"]) == 1 and CANARY not in json.dumps(shared)


def test_admission_validation(client, group, deployed):
    alice = group["alice"]
    extra = alice.post("/api/v1/requests", json={**ask("hi"), "tools": []})
    assert extra.status_code == 400 and "unknown field" in extra.json()["error"]["message"]
    long = alice.post("/api/v1/requests", json=ask("word " * 4100))  # 4100 + template + 64 output > 4096
    assert long.status_code == 422 and long.json()["error"]["code"] == "CONTEXT_TOO_LONG"
    big = alice.post("/api/v1/requests", content=b"x" * (2 << 20), headers={"Content-Type": "application/json"})
    assert big.status_code == 413


def test_idempotency_keys(client, group, deployed):
    alice = group["alice"]
    a = alice.post("/api/v1/requests", json=ask("once", client_request_id="k1")).json()
    b = alice.post("/api/v1/requests", json=ask("once", client_request_id="k1")).json()
    assert a["request_id"] == b["request_id"]
    c = alice.post("/api/v1/requests", json=ask("different", client_request_id="k1"))
    assert c.status_code == 409 and c.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_queue_limits_return_429(client, group, deployed, runtime):
    runtime.token_delay_s = 0.05
    alice = group["alice"]
    codes = [alice.post("/api/v1/requests", json=ask(f"q{i}")).status_code for i in range(4)]
    assert 429 in codes
    r = alice.post("/api/v1/requests", json=ask("again"))
    assert r.status_code == 429 and r.headers["Retry-After"] and r.json()["error"]["retryable"]


def test_cancel_running_keeps_partial_labeled_incomplete(client, group, deployed, runtime):
    runtime.token_delay_s = 0.05
    alice = group["alice"]
    admitted = alice.post("/api/v1/requests", json=ask("long answer please")).json()
    for _ in range(100):
        if alice.get(admitted["status_url"]).json()["status"] == "running":
            break
        time.sleep(0.02)
    time.sleep(0.15)
    alice.post(f"/api/v1/requests/{admitted['request_id']}/cancel")
    with client.stream("GET", admitted["events_url"], headers=alice._h()) as s:
        events = sse(s)
    assert events[-1]["type"] == "cancelled" and not any(e["type"] == "completed" for e in events)
    msgs = alice.get(f"/api/v1/conversations/{admitted['branch_id']}").json()["messages"]
    assert msgs[-1]["completion_state"] == "incomplete"


def test_worker_failure_is_never_reported_as_success(client, group, deployed, runtime):
    runtime.fail_after_tokens = 3
    _, events = run(group["alice"], ask("this will fail"))
    assert events[-1]["type"] == "failed" and events[-1]["payload"]["code"] == "WORKER_LOST"
    assert not any(e["type"] == "completed" for e in events)


def test_device_loss_marks_deployment_unavailable(client, group, deployed, runtime, reference_devices):
    runtime.token_delay_s = 0.1
    alice, gid = group["alice"], group["group"]["group_id"]
    admitted = alice.post("/api/v1/requests", json=ask("during loss")).json()
    store = client.app.state.store
    with store.tx() as c:
        c.execute("UPDATE devices SET last_seen = '2000-01-01T00:00:00+00:00' WHERE id = ?",
                  (reference_devices["vm"]["device_id"],))
    with client.stream("GET", admitted["events_url"], headers=alice._h()) as s:
        events = sse(s)
    assert events[-1]["type"] == "failed"
    assert alice.get(f"/api/v1/groups/{gid}/deployment").json()["state"] == "unavailable"
    r = alice.post("/api/v1/requests", json=ask("after loss"))
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"


# --- Devices, policy, placement ----------------------------------------------------------------------
def test_pilot_test_c_deployment_across_reference_group(client, group, reference_devices):
    alice, gid = group["alice"], group["group"]["group_id"]
    r = alice.post(f"/api/v1/groups/{gid}/deployment", json={"model": "pilot-test-c"})
    assert r.status_code == 200, r.text
    exp = r.json()["explanation"]
    assert exp["single_device_fit"] == [] and len(exp["stages"]) == 3
    assert "Test C" in exp["capacity_basis"]
    member_view = group["bob"].get(f"/api/v1/groups/{gid}/deployment").json()
    assert "explanation" not in member_view and member_view["state"] == "ready"


def test_owner_controls_limits_admin_can_only_disable(client, group, reference_devices):
    alice, bob, gid = group["alice"], group["bob"], group["group"]["group_id"]
    hp = reference_devices["hp"]["device_id"]
    raise_attempt = alice.patch(f"/api/v1/devices/{hp}/policy", {"memory_cap_bytes": 30 * GB})
    assert raise_attempt.status_code == 403
    assert bob.patch(f"/api/v1/devices/{hp}/policy", {"memory_cap_bytes": 6 * GB}).status_code == 200
    assert alice.patch(f"/api/v1/devices/{hp}/policy", {"disabled": True}).json()["status"] == "disabled"
    r = alice.post(f"/api/v1/groups/{gid}/deployment", json={"model": "pilot-test-c"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "PLACEMENT_INFEASIBLE"
    filtered = {f["device_id"]: f["filter"] for f in r.json()["error"]["explanation"]["filtered_out"]}
    assert filtered[hp] == "authorized"


def test_battery_device_is_filtered_by_default(client, group):
    alice, gid = group["alice"], group["group"]["group_id"]
    enroll(client, alice, "Mac Studio", 110, 64, 800, 1)
    lap = enroll(client, alice, "Laptop", 10.2, 8, 51, 5, power="battery")
    enroll(client, alice, "VM", 5, 4.5, 15, 60)
    r = alice.post(f"/api/v1/groups/{gid}/deployment", json={"model": "pilot-test-c"})
    assert r.status_code == 409
    assert {"device_id": lap["device_id"], "filter": "power",
            "reason": "on battery and owner allows plugged-in contribution only"} in r.json()["error"]["explanation"]["filtered_out"]


def test_enrollment_code_single_use(client, group):
    bob = group["bob"]
    code = bob.post("/api/v1/devices/enrollment-codes").json()["code"]
    body = {"code": code, "name": "x", "public_key": "k", "policy": {"memory_cap_bytes": GB},
            "capabilities": {"os": "t", "arch": "t", "backend_digest": "fake-runtime-0.1", "formats": ["gguf"],
                             "allocatable_bytes": GB, "mem_bandwidth_gbps": 10, "rtt_ms": 1}}
    assert client.post("/api/v1/devices/enroll", json=body).status_code == 201
    assert client.post("/api/v1/devices/enroll", json=body).status_code == 404


# --- Membership revocation (FR 04) ------------------------------------------------------------------
def test_member_removal_revokes_sessions_keys_and_devices(client, group, deployed, reference_devices):
    alice, bob, gid = group["alice"], group["bob"], group["group"]["group_id"]
    key = bob.post("/api/v1/credentials", json={"name": "app"}).json()["secret"]
    assert alice.delete(f"/api/v1/groups/{gid}/members/{bob.member_id}").status_code == 204
    assert bob.get("/api/v1/auth/session").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code == 401
    hp = reference_devices["hp"]
    hb = client.post(f"/api/v1/devices/{hp['device_id']}/heartbeat",
                     json={"capabilities": hp["capabilities"], "paused": False},
                     headers={"Authorization": f"Bearer {hp['device_token']}"})
    assert hb.status_code == 403
    assert alice.delete(f"/api/v1/groups/{gid}/members/{alice.member_id}").status_code == 409


# --- OpenAI-compatible subset (FR 22, FR 25) ----------------------------------------------------------
def test_compat_chat_completions(client, group, deployed):
    bob = group["bob"]
    key = bob.post("/api/v1/credentials", json={"name": "app"}).json()["secret"]
    auth = {"Authorization": f"Bearer {key}"}
    models = client.get("/v1/models", headers=auth).json()
    assert models["data"][0]["id"] == "group-default"
    body = {"model": "group-default", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}
    r = client.post("/v1/chat/completions", json=body, headers=auth)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["object"] == "chat.completion" and out["choices"][0]["message"]["content"]
    assert out["choices"][0]["finish_reason"] == "length"
    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}, headers=auth) as s:
        lines = [line for line in s.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]" and json.loads(lines[0][6:])["choices"][0]["delta"]["role"] == "assistant"


def test_compat_rejects_unsupported_fields_and_scopes(client, group, deployed):
    bob = group["bob"]
    key = bob.post("/api/v1/credentials", json={"name": "app"}).json()["secret"]
    auth = {"Authorization": f"Bearer {key}"}
    base = {"model": "group-default", "messages": [{"role": "user", "content": "hi"}]}
    tools = client.post("/v1/chat/completions", json={**base, "tools": [{}]}, headers=auth)
    assert tools.status_code == 400 and tools.json()["error"]["code"] == "UNSUPPORTED_FIELD"
    image = client.post("/v1/chat/completions", headers=auth, json={
        **base, "messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
    assert image.status_code == 400
    conv = bob.post("/api/v1/conversations", json={"visibility": "group"}).json()
    share = client.post("/v1/chat/completions", headers=auth,
                        json={**base, "difa_visibility": "group", "difa_branch_id": conv["branch_id"]})
    assert share.status_code == 403  # inference-only key cannot publish shared content
    cookie = client.post("/v1/chat/completions", json=base, headers=bob._h())
    assert cookie.status_code == 401
