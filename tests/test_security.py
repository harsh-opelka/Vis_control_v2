"""Tests for ``viscontrol.core.security``."""

from __future__ import annotations

import pytest

from viscontrol.core.security import hash_pin, verify_pin


def test_hash_and_verify_default_pin() -> None:
    hashed = hash_pin("0000")
    assert hashed != "0000"
    assert hashed.startswith("$2")  # bcrypt prefix
    assert verify_pin("0000", hashed) is True
    assert verify_pin("0001", hashed) is False


def test_verify_pin_rejects_empty_inputs() -> None:
    h = hash_pin("1234")
    assert verify_pin("", h) is False
    assert verify_pin("1234", "") is False


def test_verify_pin_rejects_malformed_hash() -> None:
    assert verify_pin("1234", "not-a-hash") is False


def test_hash_pin_is_salted() -> None:
    # Two hashes of the same PIN must differ thanks to per-call salt,
    # but both must verify.
    a = hash_pin("hunter2")
    b = hash_pin("hunter2")
    assert a != b
    assert verify_pin("hunter2", a)
    assert verify_pin("hunter2", b)


def test_hash_pin_type_check() -> None:
    with pytest.raises(TypeError):
        hash_pin(1234)  # type: ignore[arg-type]
