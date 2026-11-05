"""llama.cpp runtime adapter (Phase 1 feasibility only; spec 6.2).

The coordinator host runs llama-server with its own Metal device for the stage it owns and llama.cpp
RPC devices for remote stages. Layer counts from the placement pipeline become --tensor-split shares,
and the adapter verifies from llama.cpp's own load log that every stage actually received weights.

NOT a release backend. Upstream documents llama.cpp RPC as fragile and insecure, so this adapter only
connects to loopback or explicitly configured research endpoints, binds llama-server to 127.0.0.1 with a
random API key, and must only be used with synthetic prompts (spec 6.2).
"""
import asyncio
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
from pathlib import Path

import httpx

from ..catalog import MODEL_FILES
from .base import Cancelled, Generation, RuntimeFailure

LOOPBACK_PREFIXES = ("127.", "localhost", "[::1]")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_model_buffers(log_text: str) -> dict[str, float]:
    """MiB of weights per backend device from llama.cpp load_tensors lines; RPC keyed by endpoint."""
    out: dict[str, float] = {}
    for m in re.finditer(r"load_tensors:\s+(\S+?)(?:\[([^\]]*)\])?(?:_Mapped)? model buffer size =\s+([\d.]+) MiB",
                         log_text):
        key = f"rpc:{m.group(2)}" if m.group(2) else m.group(1)
        out[key] = round(out.get(key, 0) + float(m.group(3)), 2)
    return out


class LlamaCppRuntime:
    protocol_version = "difa-adapter/0"
    capabilities = {"offloadModes": [], "conversationCache": False, "speculativeDecode": False}

    def __init__(self, bin_dir: str, model_dir: str, local_device: str, rpc_endpoints: dict[str, str],
                 backend_digest: str = "llama.cpp-v0.9.0", allow_remote_rpc: bool = False, log_dir: str | None = None):
        self.bin = Path(bin_dir)
        self.model_dir = Path(model_dir)
        self.local_device = local_device            # device name served by this host's Metal device
        self.rpc_endpoints = dict(rpc_endpoints)    # device name -> host:port of its rpc-server
        self.backend_digest = backend_digest
        self.log_dir = Path(log_dir or Path.home() / ".cache" / "difa-phase1" / "adapter-logs")
        for name, ep in self.rpc_endpoints.items():
            if not allow_remote_rpc and not ep.startswith(LOOPBACK_PREFIXES):
                raise ValueError(f"{name}: non-loopback RPC endpoint {ep} refused; llama.cpp RPC is insecure")
        self._deployments: dict[str, dict] = {}
        self._verified: dict[str, str] = {}

    @classmethod
    def from_env(cls) -> "LlamaCppRuntime":
        return cls(bin_dir=os.environ["DIFA_LLAMACPP_BIN"], model_dir=os.environ["DIFA_MODEL_DIR"],
                   local_device=os.environ["DIFA_LLAMACPP_LOCAL_DEVICE"],
                   rpc_endpoints=json.loads(os.environ.get("DIFA_LLAMACPP_RPC", "{}")))

    def _model_path(self, name: str) -> Path:
        meta = MODEL_FILES.get(name)
        if not meta:
            raise RuntimeFailure("UNSUPPORTED_MODEL", f"{name} has no pinned model file.", retryable=False)
        path = self.model_dir / meta["file"]
        if not path.exists():
            raise RuntimeFailure("MODEL_MISSING", f"Model file {meta['file']} is not downloaded.", retryable=False)
        if self._verified.get(str(path)) != meta["sha256"]:
            if _sha256(path) != meta["sha256"]:
                raise RuntimeFailure("MODEL_HASH_MISMATCH", "Model file failed hash verification.", retryable=False)
            self._verified[str(path)] = meta["sha256"]
        return path

    async def load_deployment(self, deployment_id: str, placement: dict) -> dict:
        model_name = placement["model"]["name"]
        path = await asyncio.to_thread(self._model_path, model_name)
        devs, rpc, owners = [], [], {}
        for st in placement["stages"]:
            if st["name"] == self.local_device:
                devs.append("MTL0")
                owners["MTL0"] = st
            elif st["name"] in self.rpc_endpoints:
                ep = self.rpc_endpoints[st["name"]]
                devs.append(f"RPC{len(rpc)}")
                rpc.append(ep)
                owners[f"rpc:{ep}"] = st
            else:
                raise RuntimeFailure("NO_ENDPOINT", f"No llama.cpp endpoint configured for {st['name']}.", False)
        shares = [st["layers"][1] - st["layers"][0] for st in placement["stages"]]
        port, key = _free_port(), secrets.token_urlsafe(24)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{deployment_id}.log"
        args = [str(self.bin / "llama-server"), "-m", str(path), "--host", "127.0.0.1", "--port", str(port),
                "--api-key", key, "-c", str(placement["model"].get("max_context_tokens", 4096)), "-np", "1",
                "--no-webui", "-lv", "4", "-fit", "off", "-ngl", "99"]
        if rpc:  # must precede -dev: llama-server registers RPC devices when it parses --rpc
            args += ["--rpc", ",".join(rpc)]
        args += ["-dev", ",".join(devs), "-ts", ",".join(str(s) for s in shares)]
        log = open(log_path, "w")
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {key}"},
                                   timeout=httpx.Timeout(300, connect=5))
        try:
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeFailure("LOAD_FAILED", "The inference server exited while loading.")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
            else:
                raise RuntimeFailure("LOAD_TIMEOUT", "The inference server did not become ready.")
            buffers = parse_model_buffers(log_path.read_text(errors="replace"))
            partition_map = []
            for dev_key, st in owners.items():
                mib = buffers.get(dev_key, 0.0)
                partition_map.append({"device_id": st["device_id"], "layers": st["layers"], "model_mib": mib})
            missing = [p["device_id"] for p in partition_map if p["model_mib"] <= 0]
            if missing:
                raise RuntimeFailure("PARTITION_MISMATCH", f"Devices received no weights: {', '.join(missing)}.")
        except BaseException:
            proc.terminate()
            await client.aclose()
            raise
        order = {st["device_id"]: i for i, st in enumerate(placement["stages"])}
        partition_map.sort(key=lambda p: order[p["device_id"]])
        self._deployments[deployment_id] = {"proc": proc, "client": client, "partition_map": partition_map,
                                            "log": str(log_path), "model": model_name}
        return {"manifest_digest": MODEL_FILES[model_name]["sha256"][:16], "partition_map": partition_map,
                "backend_digest": self.backend_digest}

    def partition_map(self, deployment_id: str) -> list[dict]:
        return self._deployments[deployment_id]["partition_map"]

    async def unload_deployment(self, deployment_id: str) -> None:
        dep = self._deployments.pop(deployment_id, None)
        if dep:
            dep["proc"].terminate()
            await dep["client"].aclose()

    async def health(self) -> dict:
        return {"ok": all(d["proc"].poll() is None for d in self._deployments.values()),
                "deployments": list(self._deployments)}

    async def generate_stream(self, deployment_id, messages, max_tokens, seed, cancel: asyncio.Event,
                              result: Generation):
        dep = self._deployments.get(deployment_id)
        if not dep or dep["proc"].poll() is not None:
            raise RuntimeFailure("MODEL_UNAVAILABLE", "The inference server is not running.")
        body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0,
                "cache_prompt": False}
        if seed is not None:
            body["seed"] = seed
        try:
            async with dep["client"].stream("POST", "/v1/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    raise RuntimeFailure("BACKEND_ERROR", f"Inference server returned {resp.status_code}.")
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        raise Cancelled()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    for choice in chunk.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            result.completion_tokens += 1
                            yield piece
                        if choice.get("finish_reason"):
                            result.finish_reason = "length" if choice["finish_reason"] == "length" else "stop"
        except (httpx.HTTPError, json.JSONDecodeError):
            raise RuntimeFailure("WORKER_LOST", "The inference server or a required device stopped responding.")
        if cancel.is_set():
            raise Cancelled()
        if result.finish_reason is None:
            raise RuntimeFailure("WORKER_LOST", "Generation ended without a finish reason.")
