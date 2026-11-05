"""Request lifecycle (spec 7.1). Every admitted request reaches exactly one terminal state."""
from enum import Enum


class RequestState(str, Enum):
    VALIDATED = "validated"
    QUEUED = "queued"
    RESERVED = "reserved"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


TERMINAL = frozenset({RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED, RequestState.EXPIRED})

_ALLOWED = {
    RequestState.VALIDATED: {RequestState.QUEUED, RequestState.FAILED},
    RequestState.QUEUED: {RequestState.RESERVED, RequestState.CANCELLED, RequestState.EXPIRED, RequestState.FAILED},
    RequestState.RESERVED: {RequestState.RUNNING, RequestState.CANCELLED, RequestState.EXPIRED, RequestState.FAILED},
    RequestState.RUNNING: {RequestState.COMPLETED, RequestState.CANCELLED, RequestState.EXPIRED, RequestState.FAILED},
}


class InvalidTransition(ValueError):
    pass


def check_transition(current: RequestState, new: RequestState) -> None:
    if new not in _ALLOWED.get(current, set()):
        raise InvalidTransition(f"{current.value} -> {new.value} is not allowed")
