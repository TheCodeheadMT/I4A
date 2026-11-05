"""Per-device memory reservation ledger (FR 13, FR 15, Section 7).

Reservations are all-or-nothing across devices, so a request either holds everything it needs or
nothing; the ledger never lets reserved bytes exceed a device's budget.
"""


class InsufficientCapacity(Exception):
    def __init__(self, device_id: str, needed: int, free: int):
        super().__init__(f"{device_id}: needs {needed} bytes, {free} free")
        self.device_id, self.needed, self.free = device_id, needed, free


class Ledger:
    def __init__(self, budgets: dict[str, int] | None = None):
        self._budget: dict[str, int] = dict(budgets or {})
        self._held: dict[str, dict[str, int]] = {}  # reservation id -> device -> bytes

    def set_budget(self, device_id: str, budget: int) -> bool:
        """Returns False when the new budget is below current reservations (device must drain)."""
        self._budget[device_id] = budget
        return self.reserved(device_id) <= budget

    def reserved(self, device_id: str) -> int:
        return sum(h.get(device_id, 0) for h in self._held.values())

    def free(self, device_id: str) -> int:
        return self._budget.get(device_id, 0) - self.reserved(device_id)

    def reserve(self, reservation_id: str, demands: dict[str, int]) -> None:
        if reservation_id in self._held:
            raise ValueError("duplicate reservation")
        for device_id, need in demands.items():
            if need < 0:
                raise ValueError("negative demand")
            if need > self.free(device_id):
                raise InsufficientCapacity(device_id, need, self.free(device_id))
        self._held[reservation_id] = dict(demands)

    def release(self, reservation_id: str) -> bool:
        return self._held.pop(reservation_id, None) is not None

    def active(self) -> list[str]:
        return list(self._held)
