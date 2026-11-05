"""Declared subset of OpenAI-style chat completions (FR 22, FR 25, spec 9.2).

Differences from full compatibility are published in the README. Unsupported tool,
function, image, and audio fields are rejected explicitly, never ignored.
"""
import asyncio
import json
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..domain.access import Principal, Visibility
from ..domain.lifecycle import RequestState
from .auth import current
from .errors import ApiError
from .schemas import RequestIn

UNSUPPORTED = {"tools", "tool_choice", "functions", "function_call", "response_format", "logprobs", "top_logprobs",
               "n", "audio", "modalities", "parallel_tool_calls", "logit_bias", "prediction"}
ACCEPTED = {"model", "messages", "stream", "max_tokens", "max_completion_tokens", "temperature", "seed", "user",
            "difa_visibility", "difa_branch_id", "difa_client_request_id"}
KEEPALIVE_S = 10


def _parse(body: dict, p: Principal) -> tuple[RequestIn, bool]:
    if not isinstance(body, dict):
        raise ApiError(400, "INVALID_FIELDS", "Body must be a JSON object.")
    bad = sorted(set(body) & UNSUPPORTED)
    if bad:
        raise ApiError(400, "UNSUPPORTED_FIELD", f"Unsupported fields: {', '.join(bad)}.")
    unknown = sorted(set(body) - ACCEPTED - UNSUPPORTED)
    if unknown:
        raise ApiError(400, "INVALID_FIELDS", f"Unknown fields: {', '.join(unknown)}.")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ApiError(400, "INVALID_FIELDS", "messages must be a non-empty list.")
    for m in msgs:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            raise ApiError(400, "UNSUPPORTED_FIELD", "Only text message content is supported.")
        if set(m) - {"role", "content", "name"}:
            raise ApiError(400, "UNSUPPORTED_FIELD", "Message fields other than role and content are unsupported.")
    visibility = body.get("difa_visibility", "private")
    if visibility == "group" and not body.get("difa_branch_id"):
        raise ApiError(400, "INVALID_FIELDS", "Shared requests must name a destination difa_branch_id.")
    try:
        req = RequestIn(model=body.get("model", ""), messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
                        visibility=Visibility(visibility),
                        max_output_tokens=body.get("max_completion_tokens") or body.get("max_tokens") or 256,
                        branch_id=body.get("difa_branch_id"), temperature=body.get("temperature"),
                        seed=body.get("seed"), client_request_id=body.get("difa_client_request_id"))
    except (ValidationError, ValueError) as e:
        raise ApiError(400, "INVALID_FIELDS", "Invalid field values: " + "; ".join(
            ".".join(str(x) for x in err["loc"]) for err in getattr(e, "errors", lambda: [])()) or "invalid")
    return req, bool(body.get("stream"))


def register(app: FastAPI) -> None:
    from .app import admit_request  # local import avoids a cycle

    @app.get("/v1/models", tags=["compat"])
    async def models(request: Request, p: Principal = Depends(current)):
        g = request.app.state.scheduler.group(p.group_id)
        data = []
        if g.deployment_id:
            data.append({"id": "group-default", "object": "model", "owned_by": "difa-group",
                         "difa_model": g.placement["model"]["name"], "difa_deployment_id": g.deployment_id})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions", tags=["compat"])
    async def chat_completions(request: Request, p: Principal = Depends(current)):
        if p.via != "credential":
            raise ApiError(401, "UNAUTHENTICATED", "Use an application credential as a Bearer token.")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise ApiError(400, "INVALID_JSON", "Body is not valid JSON.")
        req, stream = _parse(body, p)
        admitted = admit_request(request, p, req)
        rid, ch = admitted["request_id"], request.app.state.scheduler.channels[admitted["request_id"]]
        created, model = int(time.time()), req.model
        if stream:
            return StreamingResponse(_stream(ch, rid, created, model), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-store", "X-DIFA-Request-Id": rid})
        deadline = request.app.state.settings.wait_deadline_s + request.app.state.settings.active_deadline_s
        text: list[str] = []

        async def collect():
            last = None
            async for ev in ch.iterate():
                if ev["type"] == "token":
                    text.append(ev["payload"]["text"])
                elif ev["type"] in ("completed", "failed", "cancelled"):
                    last = ev
            return last

        try:
            final = await asyncio.wait_for(collect(), timeout=deadline)
        except asyncio.TimeoutError:
            request.app.state.scheduler.cancel(p.group_id, rid)
            raise ApiError(504, "DEADLINE", "The request did not finish within its deadline.", True)
        if final is None or final["type"] != "completed":
            payload = (final or {}).get("payload", {})
            raise ApiError(503 if payload.get("retryable", True) else 500, payload.get("code", "CANCELLED"),
                           payload.get("message", "The request did not complete."), payload.get("retryable", True))
        usage = final["payload"]["usage"]
        return {"id": f"chatcmpl-{rid}", "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                             "finish_reason": final["payload"]["finish_reason"]}],
                "usage": usage, "difa_request_id": rid, "difa_branch_id": admitted["branch_id"]}


async def _stream(ch, rid: str, created: int, model: str):
    def chunk(delta: dict, finish=None) -> str:
        return "data: " + json.dumps({"id": f"chatcmpl-{rid}", "object": "chat.completion.chunk", "created": created,
                                      "model": model, "choices": [{"index": 0, "delta": delta,
                                                                   "finish_reason": finish}]}) + "\n\n"
    started, i = False, 0
    while True:
        if i >= len(ch.events):
            if ch.done:
                return
            if not await ch.wait_for_new(KEEPALIVE_S):
                yield ": keepalive\n\n"  # documented comment keepalive while queued
            continue
        ev = ch.events[i]
        i += 1
        if ev["type"] == "started" and not started:
            started = True
            yield chunk({"role": "assistant", "content": ""})
        elif ev["type"] == "token":
            yield chunk({"content": ev["payload"]["text"]})
        elif ev["type"] == "completed":
            yield chunk({}, ev["payload"]["finish_reason"])
            yield "data: [DONE]\n\n"
            return
        elif ev["type"] in ("failed", "cancelled"):
            p = ev["payload"]
            # Terminal error event after headers were sent; never a success marker (spec 9.3).
            yield "data: " + json.dumps({"error": {"code": p.get("code", "CANCELLED"),
                                                   "message": p.get("message", "The request was cancelled."),
                                                   "retryable": p.get("retryable", True)}}) + "\n\n"
            return


__all__ = ["register", "RequestState"]
