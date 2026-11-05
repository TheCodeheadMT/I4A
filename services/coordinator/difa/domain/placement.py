"""Filter-and-score placement pipeline with a per-token latency budget (spec FR 31, Section 7.2).

Filters remove ineligible devices with a recorded reason. Scorers rank the rest. The planner then
uses the fewest, best-scored devices whose per-device ledgers satisfy W + K + R + X <= B, assigning
contiguous layer ranges. Every decision returns a content-free explanation that can be replayed.
"""
from dataclasses import asdict, dataclass, field

PIPELINE_VERSION = "placement-pipeline/1"
GB = 10**9


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    name: str
    os: str
    arch: str
    backend_digest: str
    formats: tuple[str, ...]
    budget_bytes: int            # min(owner cap, measured allocatable) before headroom
    mem_bandwidth_gbps: float    # measured memory bandwidth, GB/s
    rtt_ms: float                # measured round trip from the coordinator
    power: str = "ac"            # "ac", "battery", or "unknown"
    allow_battery: bool = False
    ready: bool = True
    authorized: bool = True
    paused: bool = False
    stability: float = 1.0       # fraction of recent heartbeats received


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str
    fmt: str
    model_class: str             # "dense" or "moe"
    total_params_b: float
    active_params_b: float
    layers: int
    bytes_per_layer: int
    kv_bytes_per_token_per_layer: int
    runtime_overhead_bytes: int  # R per participating device
    max_context_tokens: int
    synthetic: bool = True       # fixture, not a real model

    def kv_bytes_per_layer(self, sequences: int = 1) -> int:
        return self.kv_bytes_per_token_per_layer * self.max_context_tokens * sequences


@dataclass(frozen=True)
class PlacementPolicy:
    backend_digest: str
    headroom_fraction: float = 0.10   # X: transfers, duplication, and uncertainty
    active_sequences: int = 1          # G
    require_all: bool = False          # Test A cooperation topology
    weights: dict = field(default_factory=lambda: {"bandwidth": 0.4, "latency": 0.3, "power": 0.2, "stability": 0.1})


def _f_authorized(d, m, p): return None if d.authorized else "device authorization revoked or expired"
def _f_ready(d, m, p): return None if d.ready else "device not ready (no valid heartbeat)"
def _f_paused(d, m, p): return "paused by owner" if d.paused else None
def _f_power(d, m, p):
    if d.power == "battery" and not d.allow_battery:
        return "on battery and owner allows plugged-in contribution only"
    if d.power == "unknown" and not d.allow_battery:
        return "power status unknown; battery contribution paused by default"
    return None
def _f_digest(d, m, p):
    return None if d.backend_digest == p.backend_digest else f"backend digest {d.backend_digest} differs from {p.backend_digest}"
def _f_format(d, m, p): return None if m.fmt in d.formats else f"model format {m.fmt} unsupported"
def _f_budget(d, m, p):
    return None if _layer_capacity(d, m, p) >= 1 else "budget too small for one layer after overhead and headroom"


FILTERS = [("authorized", _f_authorized), ("ready", _f_ready), ("paused", _f_paused), ("power", _f_power),
           ("backend_digest", _f_digest), ("format", _f_format), ("budget", _f_budget)]


def _usable_bytes(d: DeviceProfile, m: ModelSpec, p: PlacementPolicy) -> int:
    return int(d.budget_bytes * (1 - p.headroom_fraction)) - m.runtime_overhead_bytes


def _per_layer_bytes(m: ModelSpec, p: PlacementPolicy) -> int:
    return m.bytes_per_layer + m.kv_bytes_per_layer(p.active_sequences)


def _layer_capacity(d: DeviceProfile, m: ModelSpec, p: PlacementPolicy) -> int:
    return max(0, _usable_bytes(d, m, p) // _per_layer_bytes(m, p))


def _score(d: DeviceProfile, peers: list[DeviceProfile], w: dict) -> dict:
    max_bw = max(x.mem_bandwidth_gbps for x in peers) or 1.0
    parts = {
        "bandwidth": d.mem_bandwidth_gbps / max_bw,
        "latency": 1.0 / (1.0 + d.rtt_ms / 50.0),
        "power": {"ac": 1.0, "battery": 0.3}.get(d.power, 0.0),
        "stability": d.stability,
    }
    parts = {k: round(v, 4) for k, v in parts.items()}
    return {"total": round(sum(w[k] * v for k, v in parts.items()), 4), "components": parts}


def plan(devices: list[DeviceProfile], model: ModelSpec, policy: PlacementPolicy) -> dict:
    explanation: dict = {
        "pipeline_version": PIPELINE_VERSION,
        "model": {"name": model.name, "revision": model.revision, "class": model.model_class,
                  "layers": model.layers, "weights_bytes": model.layers * model.bytes_per_layer,
                  "synthetic": model.synthetic},
        "policy": asdict(policy),
        "filtered_out": [],
        "scores": [],
        "single_device_fit": [],
        "feasible": False,
    }
    eligible = []
    for d in devices:
        reason = next(((n, r) for n, f in FILTERS if (r := f(d, model, policy))), None)
        if reason:
            explanation["filtered_out"].append({"device_id": d.device_id, "filter": reason[0], "reason": reason[1]})
        else:
            eligible.append(d)

    scored = sorted(((d, _score(d, eligible, policy.weights)) for d in eligible),
                    key=lambda x: (-x[1]["total"], x[0].device_id))
    explanation["scores"] = [{"device_id": d.device_id, **s} for d, s in scored]
    explanation["single_device_fit"] = [d.device_id for d, _ in scored
                                        if _layer_capacity(d, model, policy) >= model.layers]
    if not scored:
        explanation["reason"] = "no eligible devices"
        return explanation

    ranked = [d for d, _ in scored]
    candidate_sets = [ranked] if policy.require_all else [ranked[:k] for k in range(1, len(ranked) + 1)]
    chosen = assignment = None
    for subset in candidate_sets:
        assignment = _assign(subset, model, policy)
        if assignment:
            chosen = subset
            break
    if not chosen:
        capacity = sum(_layer_capacity(d, model, policy) for d in ranked)
        explanation["reason"] = (f"insufficient combined capacity: {capacity} of {model.layers} layers fit "
                                 f"across {len(ranked)} eligible devices")
        return explanation

    stages, start = [], 0
    per_layer = _per_layer_bytes(model, policy)
    for d, n in zip(chosen, assignment):
        weights = n * model.bytes_per_layer
        kv = n * model.kv_bytes_per_layer(policy.active_sequences)
        need = weights + kv + model.runtime_overhead_bytes
        stages.append({
            "device_id": d.device_id, "name": d.name, "layers": [start, start + n],
            "weights_bytes": weights, "kv_bytes": kv, "runtime_bytes": model.runtime_overhead_bytes,
            "headroom_bytes": d.budget_bytes - int(d.budget_bytes * (1 - policy.headroom_fraction)),
            "budget_bytes": d.budget_bytes, "margin_bytes": _usable_bytes(d, model, policy) + model.runtime_overhead_bytes - need,
            "est_compute_ms": round(weights / (d.mem_bandwidth_gbps * GB) * 1000, 2),
            "rtt_ms": d.rtt_ms,
        })
        start += n
    assert start == model.layers and all(s["margin_bytes"] >= 0 for s in stages) and per_layer > 0

    compute_ms = sum(s["est_compute_ms"] for s in stages)
    # Coordinator-centric exchange per stage, as in an RPC backend; refine with measured traces.
    network_ms = sum(s["rtt_ms"] for s in stages)
    token_ms = compute_ms + network_ms
    explanation.update({
        "feasible": True,
        "stages": stages,
        "reservation_bytes": {s["device_id"]: s["kv_bytes"] for s in stages},
        "latency_budget": {
            "compute_ms": round(compute_ms, 2), "network_ms": round(network_ms, 2),
            "per_token_ms": round(token_ms, 2), "tokens_per_s": round(1000 / token_ms, 2) if token_ms else None,
            "dominant_stage": max(stages, key=lambda s: s["est_compute_ms"] + s["rtt_ms"])["device_id"],
            "basis": "nominal estimate: weights/bandwidth plus one round trip per stage",
        },
        "capacity_basis": ("fits a single eligible device budget" if explanation["single_device_fit"]
                           else "exceeds every eligible device budget (owner-budget capacity, Test C)"),
    })
    return explanation


def _assign(subset: list[DeviceProfile], model: ModelSpec, policy: PlacementPolicy) -> list[int] | None:
    caps = [_layer_capacity(d, model, policy) for d in subset]
    if sum(caps) < model.layers or any(c < 1 for c in caps):
        return None
    remaining = model.layers - len(subset)  # every chosen device gets at least one layer
    counts = [1] * len(subset)
    for i, cap in enumerate(caps):
        take = min(cap - 1, remaining)
        counts[i] += take
        remaining -= take
    return counts if remaining == 0 else None
