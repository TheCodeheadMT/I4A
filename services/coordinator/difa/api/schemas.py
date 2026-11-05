"""Request and response schemas. Unknown fields are rejected (spec 9.2)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain.access import Visibility


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


Passphrase = Field(min_length=15, max_length=1024)
Login = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")


class CreateGroupIn(Strict):
    name: str = Field(min_length=1, max_length=100)
    owner_login: str = Login
    owner_display_name: str = Field(min_length=1, max_length=100)
    passphrase: str = Passphrase


class LoginIn(Strict):
    group_id: str
    login: str = Login
    passphrase: str = Field(min_length=1, max_length=1024)


class RecoverIn(Strict):
    group_id: str
    login: str = Login
    recovery_code: str = Field(min_length=1, max_length=200)
    new_passphrase: str = Passphrase


class InviteIn(Strict):
    role: Literal["member", "admin"] = "member"
    ttl_hours: int = Field(default=24, ge=1, le=168)


class CodeIn(Strict):
    code: str = Field(min_length=1, max_length=200)


class RedeemIn(Strict):
    code: str = Field(min_length=1, max_length=200)
    login: str = Login
    display_name: str = Field(min_length=1, max_length=100)
    passphrase: str = Passphrase


class CredentialIn(Strict):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[Literal["inference", "share"]] = ["inference"]
    expires_days: int = Field(default=30, ge=1, le=90)


class Capabilities(Strict):
    os: str = Field(max_length=40)
    arch: str = Field(max_length=40)
    backend_digest: str = Field(max_length=100)
    formats: list[str] = Field(max_length=10)
    allocatable_bytes: int = Field(ge=0)
    mem_bandwidth_gbps: float = Field(gt=0)
    rtt_ms: float = Field(ge=0)
    power: Literal["ac", "battery", "unknown"] = "unknown"
    battery_pct: float | None = Field(default=None, ge=0, le=100)


class Policy(Strict):
    memory_cap_bytes: int = Field(ge=0)
    allow_battery: bool = False
    battery_stop_pct: float = Field(default=30, ge=0, le=100)
    battery_resume_pct: float = Field(default=40, ge=0, le=100)
    paused: bool = False
    warm_hold_minutes: int = Field(default=30, ge=0, le=1440)


class EnrollIn(Strict):
    code: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=100)
    public_key: str = Field(min_length=1, max_length=500)
    capabilities: Capabilities
    policy: Policy


class HeartbeatIn(Strict):
    capabilities: Capabilities
    paused: bool = False  # effective local state after the agent applies owner policy


class PolicyPatch(Strict):
    memory_cap_bytes: int | None = Field(default=None, ge=0)
    allow_battery: bool | None = None
    battery_stop_pct: float | None = Field(default=None, ge=0, le=100)
    battery_resume_pct: float | None = Field(default=None, ge=0, le=100)
    paused: bool | None = None
    warm_hold_minutes: int | None = Field(default=None, ge=0, le=1440)
    disabled: bool | None = None  # administrators may only disable


class DeploymentIn(Strict):
    model: str
    require_all: bool = False


class ConversationIn(Strict):
    visibility: Visibility
    title: str | None = Field(default=None, max_length=120)


class ForkIn(Strict):
    visibility: Visibility
    include_message_ids: list[str] = []
    include_all: bool = False
    preview_confirmed: bool = False
    title: str | None = Field(default=None, max_length=120)


class MessageIn(Strict):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=200_000)


class RequestIn(Strict):
    model: str
    messages: list[MessageIn] = Field(min_length=1)
    visibility: Visibility
    max_output_tokens: int = Field(ge=1, le=512)
    branch_id: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    seed: int | None = None
    client_request_id: str | None = Field(default=None, max_length=100)
