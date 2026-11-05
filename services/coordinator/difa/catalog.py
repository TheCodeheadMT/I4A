"""Allowlisted model catalog (FR 11, FR 28). Entries here are synthetic fixtures for the fake runtime.

pilot-test-c is sized for the v0.3 pilot decision (spec 3.2): it exceeds a 64 GB owner cap on the Mac
but fits the Mac, HP, and VM budgets combined. Replace with a real pinned manifest in Phase 1.
"""
from .domain.placement import GB, ModelSpec

FAKE_BACKEND_DIGEST = "fake-runtime-0.1"

CATALOG: dict[str, ModelSpec] = {
    "fixture-small": ModelSpec(
        name="fixture-small", revision="fixture-1", fmt="gguf", model_class="dense",
        total_params_b=1.0, active_params_b=1.0, layers=24, bytes_per_layer=int(0.08 * GB),
        kv_bytes_per_token_per_layer=4096, runtime_overhead_bytes=int(0.5 * GB), max_context_tokens=4096),
    "pilot-test-c": ModelSpec(
        name="pilot-test-c", revision="fixture-1", fmt="gguf", model_class="dense",
        total_params_b=70.0, active_params_b=70.0, layers=64, bytes_per_layer=int(0.95 * GB),
        kv_bytes_per_token_per_layer=4096, runtime_overhead_bytes=int(2 * GB), max_context_tokens=4096),
}
