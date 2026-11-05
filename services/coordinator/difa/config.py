"""Proposed defaults from spec v0.3. Every value here is a baseline to be revised in the decision log."""
import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: os.environ.get("DIFA_DB", "difa.sqlite3"))
    # Sessions (spec 6.5)
    session_idle_s: int = 30 * 60
    session_absolute_s: int = 12 * 3600
    min_passphrase_len: int = 15
    login_attempts_per_window: int = 10
    login_window_s: int = 15 * 60
    # Invitations and credentials (FR 02, FR 23)
    invite_ttl_hours_default: int = 24
    invite_ttl_hours_max: int = 168
    credential_days_default: int = 30
    credential_days_max: int = 90
    # Queue and deadlines (FR 24, Section 7)
    per_member_queue: int = 2
    group_queue: int = 10
    active_generations: int = 1
    wait_deadline_s: float = field(default_factory=lambda: _env_float("DIFA_WAIT_DEADLINE_S", 600))
    active_deadline_s: float = field(default_factory=lambda: _env_float("DIFA_ACTIVE_DEADLINE_S", 300))
    # Request limits (spec 9.2)
    max_body_bytes: int = 1 << 20
    max_context_tokens: int = 4096
    max_output_tokens: int = 512
    # Device liveness (spec 9.4)
    heartbeat_interval_s: float = 5
    heartbeat_unassign_s: float = 15
    # Prototype-only knobs
    fake_token_delay_s: float = field(default_factory=lambda: _env_float("DIFA_FAKE_TOKEN_DELAY", 0.02))
    argon2_fast: bool = field(default_factory=lambda: os.environ.get("DIFA_ARGON2_FAST") == "1")
    cookie_secure: bool = field(default_factory=lambda: os.environ.get("DIFA_COOKIE_SECURE", "0") == "1")
