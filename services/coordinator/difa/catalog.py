"""Allowlisted model catalog (FR 11, FR 28).

fixture-small and pilot-test-c are synthetic fixtures for the fake runtime. pilot-test-c is sized for
the v0.3 pilot decision (spec 3.2): it exceeds a 64 GB owner cap on the Mac but fits the Mac, HP, and VM
budgets combined, including a 1 GB output head on the final stage.

qwen2.5-0.5b-q8 is a real pinned manifest used by the llama.cpp adapter. Its sizes were measured from
llama.cpp v0.9.0 load logs on 2026-11-05: 24 layers of about 15.1 MiB, a 137.9 MiB output head, and
512 bytes of KV cache per token per layer.
"""
from .domain.placement import GB, ModelSpec

FAKE_BACKEND_DIGEST = "fake-runtime-0.1"
MIB = 1 << 20

CATALOG: dict[str, ModelSpec] = {
    "fixture-small": ModelSpec(
        name="fixture-small", revision="fixture-1", fmt="gguf", model_class="dense",
        total_params_b=1.0, active_params_b=1.0, layers=24, bytes_per_layer=int(0.08 * GB),
        kv_bytes_per_token_per_layer=4096, runtime_overhead_bytes=int(0.5 * GB), max_context_tokens=4096),
    "pilot-test-c": ModelSpec(
        name="pilot-test-c", revision="fixture-2", fmt="gguf", model_class="dense",
        total_params_b=70.0, active_params_b=70.0, layers=64, bytes_per_layer=int(0.93 * GB),
        kv_bytes_per_token_per_layer=4096, runtime_overhead_bytes=int(2 * GB), max_context_tokens=4096,
        head_bytes=int(1 * GB)),
    "qwen2.5-0.5b-q8": ModelSpec(
        name="qwen2.5-0.5b-q8", revision="Qwen/Qwen2.5-0.5B-Instruct-GGUF@q8_0", fmt="gguf", model_class="dense",
        total_params_b=0.63, active_params_b=0.63, layers=24, bytes_per_layer=int(15.12 * MIB),
        kv_bytes_per_token_per_layer=512, runtime_overhead_bytes=64 * MIB, max_context_tokens=4096,
        synthetic=False, head_bytes=int(137.94 * MIB)),
}

# Real model files, pinned by hash (FR 11, FR 12).
MODEL_FILES: dict[str, dict] = {
    "qwen2.5-0.5b-q8": {
        "file": "qwen2.5-0.5b-instruct-q8_0.gguf",
        "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e",
        "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "license": "apache-2.0",
    },
}
