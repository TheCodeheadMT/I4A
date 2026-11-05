import random

import pytest

from difa.catalog import CATALOG
from difa.domain.access import Branch, ForkError, Principal, Role, Scope, Visibility, can_read_branch, \
    can_write_branch, plan_fork
from difa.domain.fairqueue import FairQueue, QueueFull
from difa.domain.ledger import InsufficientCapacity, Ledger
from difa.domain.lifecycle import TERMINAL, InvalidTransition, RequestState, check_transition
from difa.domain.placement import GB, DeviceProfile, PlacementPolicy, plan

ALICE = Principal("m_alice", "g1", Role.OWNER)
BOB = Principal("m_bob", "g1", Role.MEMBER)
EVE = Principal("m_eve", "g2", Role.OWNER)


# --- Visibility (FR 17 to FR 21) -----------------------------------------------------------------
def test_private_branch_only_readable_by_owner_even_for_admins():
    private = Branch("b1", "g1", BOB.member_id, Visibility.PRIVATE)
    assert can_read_branch(BOB, private)
    assert not can_read_branch(ALICE, private)  # admin status never overrides content rules
    assert not can_read_branch(EVE, private)


def test_shared_branch_readable_by_group_but_single_writer():
    shared = Branch("b2", "g1", ALICE.member_id, Visibility.GROUP)
    assert can_read_branch(BOB, shared) and not can_write_branch(BOB, shared)
    assert can_write_branch(ALICE, shared)
    assert not can_read_branch(EVE, shared)  # other groups never cross over


def test_revoked_member_reads_nothing():
    gone = Principal("m_bob", "g1", Role.MEMBER, active=False)
    assert not can_read_branch(gone, Branch("b", "g1", "m_bob", Visibility.PRIVATE))


def test_private_to_shared_never_inherits_hidden_history():
    src = Branch("b", "g1", ALICE.member_id, Visibility.PRIVATE)
    ids = ["m1", "m2", "m3"]
    assert plan_fork(ALICE, src, Visibility.GROUP, ids, [], False, False) == []
    with pytest.raises(ForkError) as e:
        plan_fork(ALICE, src, Visibility.GROUP, ids, None, True, True)
    assert e.value.code == "INVALID_SELECTION"
    with pytest.raises(ForkError) as e:
        plan_fork(ALICE, src, Visibility.GROUP, ids, ["m2"], False, False)
    assert e.value.code == "PREVIEW_REQUIRED"
    assert plan_fork(ALICE, src, Visibility.GROUP, ids, ["m3", "m1"], False, True) == ["m1", "m3"]


def test_shared_to_private_copies_after_confirmation():
    src = Branch("b", "g1", ALICE.member_id, Visibility.GROUP)
    assert plan_fork(BOB, src, Visibility.PRIVATE, ["m1", "m2"], None, True, True) == ["m1", "m2"]
    with pytest.raises(ForkError):
        plan_fork(BOB, src, Visibility.PRIVATE, ["m1"], None, True, False)


def test_fork_rejects_foreign_messages_and_unreadable_sources():
    src = Branch("b", "g1", ALICE.member_id, Visibility.PRIVATE)
    with pytest.raises(ForkError) as e:
        plan_fork(BOB, src, Visibility.PRIVATE, ["m1"], ["m1"], False, True)
    assert e.value.code == "NOT_FOUND"
    with pytest.raises(ForkError):
        plan_fork(ALICE, src, Visibility.PRIVATE, ["m1"], ["mX"], False, True)


def test_credential_without_share_scope_cannot_publish():
    key = Principal("m_bob", "g1", Role.MEMBER, via="credential", scopes=frozenset({Scope.INFERENCE}))
    with pytest.raises(ForkError) as e:
        plan_fork(key, Branch("b", "g1", "m_bob", Visibility.PRIVATE), Visibility.GROUP, [], [], False, False)
    assert e.value.code == "FORBIDDEN"


# --- Lifecycle (spec 7.1) --------------------------------------------------------------------------
def test_terminal_states_are_final():
    for t in TERMINAL:
        for s in RequestState:
            with pytest.raises(InvalidTransition):
                check_transition(t, s)


def test_cannot_skip_reservation():
    with pytest.raises(InvalidTransition):
        check_transition(RequestState.QUEUED, RequestState.RUNNING)
    check_transition(RequestState.QUEUED, RequestState.RESERVED)
    check_transition(RequestState.RESERVED, RequestState.RUNNING)


# --- Fair queue (FR 24) ------------------------------------------------------------------------------
def test_round_robin_across_members_fifo_within_member():
    q = FairQueue(per_member=3, group_limit=10)
    for rid in ("a1", "a2", "a3"):
        q.push("alice", rid)
    q.push("bob", "b1")
    q.push("carol", "c1")
    assert q.service_order() == ["a1", "b1", "c1", "a2", "a3"]
    assert [q.pop()[1] for _ in range(5)] == ["a1", "b1", "c1", "a2", "a3"]
    assert q.pop() is None


def test_queue_limits_per_member_and_group():
    q = FairQueue(per_member=2, group_limit=3)
    q.push("alice", "a1")
    q.push("alice", "a2")
    with pytest.raises(QueueFull) as e:
        q.push("alice", "a3")
    assert e.value.scope == "member"
    q.push("bob", "b1")
    with pytest.raises(QueueFull) as e:
        q.push("carol", "c1")
    assert e.value.scope == "group"


def test_remove_and_position():
    q = FairQueue()
    q.push("alice", "a1")
    q.push("bob", "b1")
    assert q.position("b1") == 2
    assert q.remove("a1") and not q.remove("a1")
    assert q.position("b1") == 1 and len(q) == 1


# --- Ledger (FR 15) ----------------------------------------------------------------------------------
def test_reservations_are_all_or_nothing():
    led = Ledger({"mac": 100, "hp": 10})
    led.reserve("r1", {"mac": 60, "hp": 5})
    with pytest.raises(InsufficientCapacity):
        led.reserve("r2", {"mac": 30, "hp": 6})
    assert led.free("mac") == 40  # failed reservation left nothing behind
    led.release("r1")
    assert led.free("hp") == 10


def test_ledger_never_overcommits_under_random_load():
    rng = random.Random(7)
    budgets = {"a": 1000, "b": 300, "c": 50}
    led, held = Ledger(budgets), []
    for i in range(2000):
        if held and rng.random() < 0.4:
            led.release(held.pop(rng.randrange(len(held))))
            continue
        demand = {d: rng.randint(0, 120) for d in budgets}
        try:
            led.reserve(f"r{i}", demand)
            held.append(f"r{i}")
        except InsufficientCapacity:
            pass
        assert all(led.reserved(d) <= budgets[d] for d in budgets)


def test_lowering_budget_below_reservations_requires_drain():
    led = Ledger({"hp": 10})
    led.reserve("r", {"hp": 8})
    assert led.set_budget("hp", 6) is False


# --- Placement pipeline (FR 31, spec 3.2, 7.2) ---------------------------------------------------------
def dev(i, budget_gb, bw, rtt, **kw):
    return DeviceProfile(device_id=i, name=i, os="t", arch="t", backend_digest=kw.pop("backend_digest", "fake-runtime-0.1"),
                         formats=("gguf",), budget_bytes=int(budget_gb * GB), mem_bandwidth_gbps=bw, rtt_ms=rtt, **kw)


REF = [dev("mac", 64, 800, 1), dev("hp", 8, 51, 5), dev("vm", 4.5, 15, 60)]
POLICY = PlacementPolicy(backend_digest="fake-runtime-0.1")


def test_test_c_pilot_needs_all_three_and_no_single_device_fits():
    out = plan(REF, CATALOG["pilot-test-c"], POLICY)
    assert out["feasible"], out.get("reason")
    assert out["single_device_fit"] == []
    assert [s["device_id"] for s in out["stages"]] == ["mac", "hp", "vm"]
    assert out["stages"][-1]["layers"][1] == 64
    assert all(s["margin_bytes"] >= 0 for s in out["stages"])
    assert "Test C" in out["capacity_basis"]
    # Latency budget: the slow VM stage dominates even though it holds few layers.
    assert out["latency_budget"]["dominant_stage"] == "vm"


def test_small_model_uses_fewest_devices_unless_cooperation_test():
    small = CATALOG["fixture-small"]
    assert [s["device_id"] for s in plan(REF, small, POLICY)["stages"]] == ["mac"]
    coop = PlacementPolicy(backend_digest="fake-runtime-0.1", require_all=True)
    assert len(plan(REF, small, coop)["stages"]) == 3


def test_filters_record_reasons_and_never_place_ineligible_devices():
    devices = [dev("mac", 64, 800, 1), dev("hp", 8, 51, 5, power="battery"),
               dev("vm", 4.5, 15, 60, backend_digest="other"), dev("old", 8, 20, 5, ready=False),
               dev("tiny", 1, 20, 5)]
    out = plan(devices, CATALOG["pilot-test-c"], POLICY)
    reasons = {f["device_id"]: f["filter"] for f in out["filtered_out"]}
    assert reasons == {"hp": "power", "vm": "backend_digest", "old": "ready", "tiny": "budget"}
    assert not out["feasible"] and "insufficient combined capacity" in out["reason"]


def test_battery_device_allowed_when_owner_opts_in():
    devices = [dev("mac", 64, 800, 1), dev("hp", 8, 51, 5, power="battery", allow_battery=True), dev("vm", 4.5, 15, 60)]
    assert plan(devices, CATALOG["pilot-test-c"], POLICY)["feasible"]


def test_plan_is_reproducible():
    a = plan(REF, CATALOG["pilot-test-c"], POLICY)
    b = plan(list(reversed(REF)), CATALOG["pilot-test-c"], POLICY)
    assert a == b
