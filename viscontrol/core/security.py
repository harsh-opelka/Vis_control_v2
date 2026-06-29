"""PIN / password hashing.

Used both for the SERVICE PIN (default ``0000``) and for the optional web
dashboard password. Both are stored as bcrypt hashes in the config — never
plaintext — so a stolen ``local.yaml`` is not immediately a credentials leak.
"""

from __future__ import annotations

import bcrypt

# Cost factor 10 keeps verify() snappy on a Jetson while still adding meaningful
# work for an attacker.
_BCRYPT_COST = 10


def hash_pin(pin: str) -> str:
    """Hash a PIN/password and return a bcrypt string (UTF-8 decoded)."""
    if not isinstance(pin, str):
        raise TypeError("pin must be a str")
    salt = bcrypt.gensalt(rounds=_BCRYPT_COST)
    return bcrypt.hashpw(pin.encode("utf-8"), salt).decode("utf-8")


def verify_pin(pin: str, hashed: str) -> bool:
    """Constant-time compare ``pin`` against a stored bcrypt hash.

    Returns False (rather than raising) on an empty or malformed hash so callers
    can treat "no PIN configured" as "verification fails closed".
    """
    if not pin or not hashed:
        return False
    try:
        return bcrypt.checkpw(pin.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
