"""Authentication: opaque server-side browser sessions with CSRF, group-bound bearer credentials,
and device tokens (spec 6.5, FR 23). Authorization is always derived from stored membership."""
from fastapi import Request

from ..domain.access import Principal, Role, Scope
from ..security import matches, verifier
from ..store import iso, now, parse
from .errors import ApiError

SESSION_COOKIE = "difa_session"
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def _unauthorized() -> ApiError:
    return ApiError(401, "UNAUTHENTICATED", "Sign in or provide a valid credential.")


def _member_principal(store, member_id: str, via: str, scopes: frozenset) -> Principal:
    m = store.one("SELECT * FROM members WHERE id = ?", (member_id,))
    if not m or m["status"] != "active":
        raise _unauthorized()
    return Principal(member_id=m["id"], group_id=m["group_id"], role=Role(m["role"]), active=True,
                     via=via, scopes=scopes)


def bearer(request: Request) -> str | None:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else None


async def current(request: Request) -> Principal:
    store, settings = request.app.state.store, request.app.state.settings
    token = bearer(request)
    if token:
        cred = store.one("SELECT * FROM credentials WHERE verifier = ?", (verifier(token),))
        if not cred or cred["revoked_at"] or parse(cred["expires_at"]) <= now():
            raise _unauthorized()
        scopes = frozenset(Scope(s) for s in cred["scopes"].split(","))
        return _member_principal(store, cred["member_id"], "credential", scopes)

    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        raise _unauthorized()
    s = store.one("SELECT * FROM sessions WHERE verifier = ?", (verifier(sid),))
    t = now()
    if (not s or (t - parse(s["last_seen"])).total_seconds() > settings.session_idle_s
            or (t - parse(s["created_at"])).total_seconds() > settings.session_absolute_s):
        if s:
            with store.tx() as c:
                c.execute("DELETE FROM sessions WHERE verifier = ?", (s["verifier"],))
        raise _unauthorized()
    if request.method in UNSAFE and not matches(request.headers.get("x-csrf-token", ""), verifier(s["csrf"])):
        raise ApiError(403, "CSRF", "Missing or invalid CSRF token.")
    with store.tx() as c:
        c.execute("UPDATE sessions SET last_seen = ? WHERE verifier = ?", (iso(t), s["verifier"]))
    return _member_principal(store, s["member_id"], "session", frozenset({Scope.INFERENCE, Scope.SHARE}))


def require_admin(p: Principal) -> None:
    if not p.is_admin:
        raise ApiError(403, "FORBIDDEN", "Only group administrators can do this.")


def require_session(p: Principal) -> None:
    if p.via != "session":
        raise ApiError(403, "FORBIDDEN", "This action requires a signed-in session.")


def require_group(p: Principal, group_id: str) -> None:
    if p.group_id != group_id:
        raise ApiError(404, "NOT_FOUND", "Not found.")
