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
    # Execution order ends on the strongest device, which also holds the output head.
    assert [s["device_id"] for s in out["stages"]] == ["vm", "hp", "mac"]
    assert out["stages"][-1]["layers"][1] == 64 and out["stages"][-1]["holds_output_head"]
    assert not any(s["holds_output_head"] for s in out["stages"][:-1])
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


def test_output_head_counts_against_the_final_stage():
    from dataclasses import replace
    model = CATALOG["pilot-test-c"]
    without = plan(REF, replace(model, head_bytes=0), POLICY)
    with_head = plan(REF, model, POLICY)
    assert with_head["stages"][-1]["weights_bytes"] - without["stages"][-1]["weights_bytes"] >= 0
    huge = replace(model, head_bytes=int(5 * GB))
    assert not plan(REF, huge, POLICY)["feasible"]  # the head alone can close the narrow Test C window


def test_real_small_model_fits_every_device_so_it_is_not_a_capacity_result():
    out = plan(REF, CATALOG["qwen2.5-0.5b-q8"], POLICY)
    assert set(out["single_device_fit"]) == {"mac", "hp", "vm"}
    assert [s["device_id"] for s in out["stages"]] == ["mac"]
    assert out["capacity_basis"] == "fits a single eligible device budget"


def test_latency_budget_counts_overhead_round_trips_and_link_traffic():
    from difa.domain.placement import LLAMACPP_RPC_V090 as cost
    out = plan(REF, CATALOG["pilot-test-c"], POLICY)
    lb = out["latency_budget"]
    assert lb["remote_stages"] == 2  # the Mac runs last and is the coordinator-local stage
    assert lb["backend_overhead_ms"] == round(cost.first_remote_ms + cost.extra_remote_ms, 2)
    assert lb["network_ms"] == round(cost.round_trips_per_token * (60 + 5), 2)  # vm and hp round trips
    slow = [dev("mac", 64, 800, 1), dev("hp", 8, 51, 5, link_mbps=25), dev("vm", 4.5, 15, 60, link_mbps=25)]
    slow_lb = plan(slow, CATALOG["pilot-test-c"], POLICY)["latency_budget"]
    traffic_ms = 2 * cost.bytes_per_token_per_remote * 8 / 25e6 * 1000
    assert abs(slow_lb["network_ms"] - (lb["network_ms"] + traffic_ms)) < 0.05


def test_committed_calibration_meets_the_30_percent_target():
    """Evidence check: held-out topologies in the committed Phase 1 run stay within spec 7.2's target."""
    import json
    from pathlib import Path
    runs = sorted((Path(__file__).resolve().parents[2] / "benchmarks" / "results").glob("phase1-latency-*.json"))
    if not runs:
        pytest.skip("no committed latency calibration run")
    report = json.loads(runs[-1].read_text())
    assert report["validation"] and all(abs(v["error_pct"]) <= 30 for v in report["validation"])
    assert all(abs(v["error_pct"]) > 30 for v in report["bandwidth_only_model"])  # the old model fails
