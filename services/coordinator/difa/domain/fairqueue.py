"""Bounded fair queue: round-robin across members, FIFO within a member (FR 24, Section 7)."""
from collections import OrderedDict, deque


class QueueFull(Exception):
    def __init__(self, scope: str):
        super().__init__(scope)
        self.scope = scope  # "member" or "group"


class FairQueue:
    def __init__(self, per_member: int = 2, group_limit: int = 10):
        self.per_member = per_member
        self.group_limit = group_limit
        self._by_member: "OrderedDict[str, deque[str]]" = OrderedDict()  # order = round-robin order
        self._owner: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._owner)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._owner

    def check(self, member_id: str) -> None:
        if len(self) >= self.group_limit:
            raise QueueFull("group")
        if len(self._by_member.get(member_id, ())) >= self.per_member:
            raise QueueFull("member")

    def push(self, member_id: str, request_id: str) -> None:
        self.check(member_id)
        self._by_member.setdefault(member_id, deque()).append(request_id)
        self._owner[request_id] = member_id

    def pop(self) -> tuple[str, str] | None:
        if not self._by_member:
            return None
        member_id, items = next(iter(self._by_member.items()))
        request_id = items.popleft()
        del self._owner[request_id]
        self._by_member.pop(member_id)
        if items:
            self._by_member[member_id] = items  # move to the back of the round
        return member_id, request_id

    def remove(self, request_id: str) -> bool:
        member_id = self._owner.pop(request_id, None)
        if member_id is None:
            return False
        items = self._by_member[member_id]
        items.remove(request_id)
        if not items:
            del self._by_member[member_id]
        return True

    def requests_of(self, member_id: str) -> list[str]:
        return list(self._by_member.get(member_id, ()))

    def service_order(self) -> list[str]:
        """Projected order in which queued requests will be served."""
        lanes = [deque(items) for items in self._by_member.values()]
        order: list[str] = []
        while lanes:
            nxt = []
            for lane in lanes:
                order.append(lane.popleft())
                if lane:
                    nxt.append(lane)
            lanes = nxt
        return order

    def position(self, request_id: str) -> int | None:
        order = self.service_order()
        return order.index(request_id) + 1 if request_id in order else None
