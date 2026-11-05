"""Per-group admission, fair queueing, reservation, and execution (spec Section 7, FR 15, FR 24).

v1 serializes generation (G = 1 per group) and reserves KV cache for the maximum configured context
on every stage before generation starts. Events are kept in a bounded in-memory buffer per request
and delivered only to authorized subscribers by the API layer.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field

from .config import Settings
from .domain.fairqueue import FairQueue
from .domain.ledger import InsufficientCapacity, Ledger
from .domain.lifecycle import TERMINAL, RequestState, check_transition
from .runtime.base import Cancelled, Generation, RuntimeFailure
from .security import new_id
from .store import Store
from .tokens import context_tokens

MAX_EVENTS = 4096


class Channel:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.events: list[dict] = []
        self.done = False
        self._signal = asyncio.Event()

    def emit(self, type_: str, payload: dict | None = None, terminal: bool = False) -> None:
        if len(self.events) < MAX_EVENTS or terminal:
            self.events.append({"request_id": self.request_id, "sequence": len(self.events) + 1,
                                "type": type_, "payload": payload or {}})
        self.done = self.done or terminal
        old, self._signal = self._signal, asyncio.Event()
        old.set()

    async def wait_for_new(self, timeout: float) -> bool:
        """Wait until another event is emitted; False on timeout. Safe to cancel."""
        try:
            await asyncio.wait_for(asyncio.shield(self._signal.wait()), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def iterate(self, after: int = 0):
        i = after
        while True:
            while i < len(self.events):
                yield self.events[i]
                i += 1
            if self.done:
                return
            signal = self._signal
            await signal.wait()


@dataclass
class GroupRuntime:
    queue: FairQueue
    ledger: Ledger = field(default_factory=Ledger)
    deployment_id: str | None = None
    placement: dict | None = None
    running: dict = field(default_factory=dict)      # request id -> cancel event
    admitted_at: dict = field(default_factory=dict)  # request id -> monotonic seconds
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    lost: set = field(default_factory=set)           # running requests interrupted by device loss


class Scheduler:
    def __init__(self, store: Store, runtime, settings: Settings):
        self.store, self.runtime, self.settings = store, runtime, settings
        self.groups: dict[str, GroupRuntime] = {}
        self.channels: dict[str, Channel] = {}
        self._tasks: set[asyncio.Task] = set()
        self._loop_task: asyncio.Task | None = None
        self._wake = asyncio.Event()

    # --- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        self._loop_task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        for g in self.groups.values():
            for ev in g.running.values():
                ev.set()
        if self._loop_task:
            self._loop_task.cancel()
        for t in list(self._tasks):
            try:
                await asyncio.wait_for(t, 2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    def group(self, group_id: str) -> GroupRuntime:
        if group_id not in self.groups:
            self.groups[group_id] = GroupRuntime(FairQueue(self.settings.per_member_queue, self.settings.group_queue))
        return self.groups[group_id]

    # --- deployments ------------------------------------------------------------------------
    def set_deployment(self, group_id: str, deployment_id: str | None, placement: dict | None) -> None:
        g = self.group(group_id)
        g.deployment_id, g.placement = deployment_id, placement
        g.ledger = Ledger({s["device_id"]: s["kv_bytes"] for s in placement["stages"]} if placement else {})

    def deployment_ready(self, group_id: str) -> bool:
        return self.group(group_id).deployment_id is not None

    # --- admission --------------------------------------------------------------------------
    def check_capacity(self, group_id: str, member_id: str) -> None:
        self.group(group_id).queue.check(member_id)

    def admit(self, group_id: str, member_id: str, request_id: str) -> None:
        g = self.group(group_id)
        g.queue.push(member_id, request_id)
        g.admitted_at[request_id] = time.monotonic()
        self.channels[request_id] = ch = Channel(request_id)
        ch.emit("queued", {"position": g.queue.position(request_id)})
        self._wake.set()

    def position(self, group_id: str, request_id: str) -> int | None:
        return self.group(group_id).queue.position(request_id)

    def cancel(self, group_id: str, request_id: str) -> bool:
        g = self.group(group_id)
        if g.queue.remove(request_id):
            self._finish(g, request_id, RequestState.CANCELLED, "cancelled", {"reason": "cancelled before start"})
            return True
        ev = g.running.get(request_id)
        if ev:
            ev.set()
            return True
        return False

    def device_lost(self, group_id: str) -> None:
        """A required partition disappeared: fail active work clearly; never claim success (spec 10)."""
        g = self.group(group_id)
        for rid, ev in g.running.items():
            g.lost.add(rid)
            ev.set()

    def cancel_member(self, group_id: str, member_id: str) -> int:
        g = self.group(group_id)
        ids = g.queue.requests_of(member_id)
        rows = self.store.all("SELECT id FROM requests WHERE group_id = ? AND member_id = ? AND status IN ('reserved','running')",
                              (group_id, member_id))
        ids += [r["id"] for r in rows]
        return sum(self.cancel(group_id, rid) for rid in ids)

    # --- internals ---------------------------------------------------------------------------
    def _transition(self, request_id: str, new: RequestState, **fields) -> None:
        row = self.store.one("SELECT status FROM requests WHERE id = ?", (request_id,))
        check_transition(RequestState(row["status"]), new)
        self.store.set_request(request_id, status=new.value, **fields)

    def _finish(self, g: GroupRuntime, request_id: str, state: RequestState, event: str, payload: dict, **fields):
        self._transition(request_id, state, **fields)
        g.admitted_at.pop(request_id, None)
        ch = self.channels.get(request_id)
        if ch:
            ch.emit(event, payload, terminal=True)

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            for g in list(self.groups.values()):
                self._expire_waiting(g)
                self._dispatch(g)

    def _expire_waiting(self, g: GroupRuntime) -> None:
        limit = time.monotonic() - self.settings.wait_deadline_s
        for rid in [r for r in g.queue.service_order() if g.admitted_at.get(r, limit + 1) < limit]:
            g.queue.remove(rid)
            self._finish(g, rid, RequestState.EXPIRED, "failed",
                         {"code": "WAIT_DEADLINE", "message": "The request waited too long and was not started.",
                          "retryable": True}, error_code="WAIT_DEADLINE")

    def _dispatch(self, g: GroupRuntime) -> None:
        while len(g.running) < self.settings.active_generations and len(g.queue):
            _, rid = g.queue.pop()
            cancel = asyncio.Event()
            g.running[rid] = cancel
            task = asyncio.get_running_loop().create_task(self._execute(g, rid, cancel))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _execute(self, g: GroupRuntime, rid: str, cancel: asyncio.Event) -> None:
        ch = self.channels[rid]
        req = self.store.one("SELECT * FROM requests WHERE id = ?", (rid,))
        parts: list[str] = []
        try:
            if req["deployment_id"] != g.deployment_id or g.placement is None:
                raise RuntimeFailure("MODEL_UNAVAILABLE", "The model is not ready. Try again when it is.")
            try:
                g.ledger.reserve(rid, g.placement["reservation_bytes"])
            except InsufficientCapacity:
                raise RuntimeFailure("CAPACITY", "Not enough reserved memory is available right now.")
            self._transition(rid, RequestState.RESERVED)
            self._transition(rid, RequestState.RUNNING)
            ch.emit("started", {"deployment_id": g.deployment_id, "attempt": req["attempt"],
                                "partition_map": self.runtime.partition_map(g.deployment_id)})
            ids = json.loads(req["context_message_ids"])
            rows = {r["id"]: r for r in self.store.all(
                f"SELECT * FROM messages WHERE id IN ({','.join('?' * len(ids))})", ids)} if ids else {}
            messages = [{"role": rows[i]["role"], "content": rows[i]["content"]} for i in ids]
            result = Generation()

            async def consume():
                async for piece in self.runtime.generate_stream(g.deployment_id, messages, req["max_output_tokens"],
                                                                req["seed"], cancel, result):
                    parts.append(piece)
                    ch.emit("token", {"text": piece})

            await asyncio.wait_for(consume(), timeout=self.settings.active_deadline_s)
            self._save_reply(req, parts, "complete")
            usage = {"prompt_tokens": context_tokens(messages), "completion_tokens": result.completion_tokens}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            self._finish(g, rid, RequestState.COMPLETED, "completed",
                         {"finish_reason": result.finish_reason, "usage": usage},
                         finish_reason=result.finish_reason, prompt_tokens=usage["prompt_tokens"],
                         completion_tokens=usage["completion_tokens"])
        except Cancelled:
            self._save_reply(req, parts, "incomplete")
            if rid in g.lost:
                self._finish(g, rid, RequestState.FAILED, "failed",
                             {"code": "WORKER_LOST", "message": "A required device became unavailable.",
                              "retryable": True}, error_code="WORKER_LOST")
            else:
                self._finish(g, rid, RequestState.CANCELLED, "cancelled", {"partial": bool(parts)})
        except asyncio.TimeoutError:
            self._save_reply(req, parts, "incomplete")
            self._finish(g, rid, RequestState.EXPIRED, "failed",
                         {"code": "ACTIVE_DEADLINE", "message": "Generation exceeded its deadline.", "retryable": True},
                         error_code="ACTIVE_DEADLINE")
        except RuntimeFailure as e:
            self._save_reply(req, parts, "incomplete")
            self._finish(g, rid, RequestState.FAILED, "failed",
                         {"code": e.code, "message": e.message, "retryable": e.retryable}, error_code=e.code)
        finally:
            g.ledger.release(rid)
            g.running.pop(rid, None)
            g.lost.discard(rid)
            self._wake.set()

    def _save_reply(self, req, parts: list[str], state: str) -> None:
        if not parts:
            return
        with self.store.tx() as c:
            self.store.add_message(c, new_id("msg"), req["branch_id"], None, "assistant", "".join(parts), state)


__all__ = ["Scheduler", "Channel", "TERMINAL"]
