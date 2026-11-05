"""Roles, principals, and content-visibility rules (spec Section 4, FR 17 to FR 21, Section 8.1).

Admin status never overrides a content rule. Every check takes an authenticated Principal built
server-side; client-submitted role claims are never consulted.
"""
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


class Visibility(str, Enum):
    PRIVATE = "private"
    GROUP = "group"


class Scope(str, Enum):
    INFERENCE = "inference"
    SHARE = "share"


@dataclass(frozen=True)
class Principal:
    member_id: str
    group_id: str
    role: Role
    active: bool = True
    via: str = "session"  # "session" or "credential"
    scopes: frozenset = field(default_factory=lambda: frozenset({Scope.INFERENCE, Scope.SHARE}))

    @property
    def is_admin(self) -> bool:
        return self.role in (Role.OWNER, Role.ADMIN)

    @property
    def is_owner(self) -> bool:
        return self.role is Role.OWNER


@dataclass(frozen=True)
class Branch:
    id: str
    group_id: str
    owner_id: str
    visibility: Visibility


def can_read_branch(p: Principal, b: Branch) -> bool:
    if not p.active or p.group_id != b.group_id:
        return False
    if b.visibility is Visibility.PRIVATE:
        return p.member_id == b.owner_id
    return True


def can_write_branch(p: Principal, b: Branch) -> bool:
    """Shared branches are single-writer in v1 (FR 21); other members fork."""
    return can_read_branch(p, b) and p.member_id == b.owner_id


def can_delete_branch(p: Principal, b: Branch) -> bool:
    return can_write_branch(p, b)


def can_create_with_visibility(p: Principal, visibility: Visibility) -> bool:
    """Credentials need an explicit share scope to create shared content (FR 23)."""
    if Scope.INFERENCE not in p.scopes:
        return False
    return visibility is Visibility.PRIVATE or Scope.SHARE in p.scopes


class ForkError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def plan_fork(p: Principal, source: Branch, target: Visibility, source_message_ids: list[str],
              selected: list[str] | None, include_all: bool, preview_confirmed: bool) -> list[str]:
    """Return the message ids to copy into a new branch (FR 18).

    Private to shared starts empty unless the user selects messages and confirms a preview, because
    hidden private context could leak through the shared answer. Shared to private may copy context
    after confirmation.
    """
    if not can_read_branch(p, source):
        raise ForkError("NOT_FOUND", "Conversation not found.")
    if not can_create_with_visibility(p, target):
        raise ForkError("FORBIDDEN", "This credential cannot create shared conversations.")
    known = set(source_message_ids)
    if selected and not set(selected) <= known:
        raise ForkError("INVALID_SELECTION", "Selected messages must belong to the source conversation.")

    widening = source.visibility is Visibility.PRIVATE and target is Visibility.GROUP
    if widening:
        if include_all:
            raise ForkError("INVALID_SELECTION",
                            "A shared branch cannot inherit all private history; select messages to publish.")
        chosen = list(selected or [])
    else:
        chosen = list(source_message_ids) if include_all else list(selected or [])

    if chosen and not preview_confirmed:
        raise ForkError("PREVIEW_REQUIRED", "Confirm the preview of the messages that will be copied.")
    order = {mid: i for i, mid in enumerate(source_message_ids)}
    return sorted(chosen, key=order.__getitem__)
