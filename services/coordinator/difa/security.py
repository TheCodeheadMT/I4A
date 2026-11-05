"""Secrets, verifiers, and passphrase hashing.

High-entropy secrets (invitations, API keys, sessions, recovery codes) are stored only as SHA-256
verifiers, which is adequate because they cannot be guessed. Human passphrases use Argon2id.
"""
import hashlib
import hmac
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def new_secret(nbytes: int = 24) -> str:
    """At least 128 bits of entropy (FR 02); default 192."""
    return secrets.token_urlsafe(nbytes)


def verifier(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def matches(secret: str, stored_verifier: str) -> bool:
    return hmac.compare_digest(verifier(secret), stored_verifier)


def fingerprint(key: bytes) -> str:
    digest = hashlib.sha256(key).hexdigest().upper()[:24]
    return "-".join(digest[i:i + 4] for i in range(0, 24, 4))


class Passwords:
    def __init__(self, fast: bool = False):
        # fast mode is for tests only; production uses argon2-cffi's Argon2id defaults.
        self._ph = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1) if fast else PasswordHasher()
        self._dummy = self._ph.hash("dummy passphrase for timing equalisation")

    def hash(self, passphrase: str) -> str:
        return self._ph.hash(passphrase)

    def verify(self, stored_hash: str | None, passphrase: str) -> bool:
        try:
            return self._ph.verify(stored_hash or self._dummy, passphrase) and stored_hash is not None
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
