"""Deterministic fake runtime. It exercises the product contracts; it is NOT evidence of distributed inference.

Output names the placement it "ran" on so partition evidence is visible end to end. Test hooks allow
injecting a worker failure after N tokens to prove there is no false success (AT 11).
"""
import asyncio
import hashlib

from ..tokens import count_tokens
from .base import Cancelled, Generation, RuntimeFailure


class FakeRuntime:
    protocol_version = "difa-adapter/0"
    backend_digest = "fake-runtime-0.1"
    capabilities = {"offloadModes": [], "conversationCache": False, "speculativeDecode": False}

    def __init__(self, token_delay_s: float = 0.02):
        self.token_delay_s = token_delay_s
        self.fail_after_tokens: int | None = None
        self._loaded: dict[str, dict] = {}

    async def load_deployment(self, deployment_id: str, placement: dict) -> dict:
        partition_map = [{"device_id": s["device_id"], "layers": s["layers"]} for s in placement["stages"]]
        digest = hashlib.sha256(repr((placement["model"], partition_map)).encode()).hexdigest()[:16]
        self._loaded[deployment_id] = {"partition_map": partition_map, "names": [s["name"] for s in placement["stages"]]}
        return {"manifest_digest": digest, "partition_map": partition_map}

    async def unload_deployment(self, deployment_id: str) -> None:
        self._loaded.pop(deployment_id, None)

    async def health(self) -> dict:
        return {"ok": True, "deployments": list(self._loaded)}

    def partition_map(self, deployment_id: str) -> list[dict]:
        return self._loaded[deployment_id]["partition_map"]

    async def generate_stream(self, deployment_id, messages, max_tokens, seed, cancel: asyncio.Event,
                              result: Generation):
        if deployment_id not in self._loaded:
            raise RuntimeFailure("MODEL_UNAVAILABLE", "The deployment is not loaded.")
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        stages = ", ".join(f"{n} layers {p['layers'][0]}-{p['layers'][1] - 1}"
                           for n, p in zip(self._loaded[deployment_id]["names"], self.partition_map(deployment_id)))
        text = (f"Simulated answer from the DIFA fake runtime. Your message had {count_tokens(last)} tokens "
                f"and {len(messages)} context messages. Placement: {stages}.")
        words = text.split(" ")
        for i, word in enumerate(words):
            if i >= max_tokens:
                result.finish_reason = "length"
                return
            if cancel.is_set():
                raise Cancelled()
            if self.fail_after_tokens is not None and i >= self.fail_after_tokens:
                raise RuntimeFailure("WORKER_LOST", "A required device stopped responding.")
            await asyncio.sleep(self.token_delay_s)
            if cancel.is_set():
                raise Cancelled()
            result.completion_tokens += 1
            yield word if i == 0 else " " + word
        result.finish_reason = "stop"
