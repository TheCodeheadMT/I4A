"""Internal adapter contract (spec 9.4). A real backend (llama.cpp, Petals, or owned) implements this."""
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


class RuntimeFailure(Exception):
    def __init__(self, code: str, message: str, retryable: bool = True):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


class Cancelled(Exception):
    pass


@dataclass
class Generation:
    finish_reason: str | None = None   # "stop" or "length"
    completion_tokens: int = 0


class RuntimeAdapter(Protocol):
    protocol_version: str
    backend_digest: str
    capabilities: dict  # offloadModes, conversationCache, speculativeDecode (spec 9.4, v0.3)

    async def load_deployment(self, deployment_id: str, placement: dict) -> dict:
        """Returns an immutable manifest digest and the actual partition map."""

    async def unload_deployment(self, deployment_id: str) -> None: ...

    def generate_stream(self, deployment_id: str, messages: list[dict], max_tokens: int, seed: int | None,
                        cancel: asyncio.Event, result: Generation) -> AsyncIterator[str]: ...

    async def health(self) -> dict: ...
