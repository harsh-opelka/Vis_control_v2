"""Tests for ``viscontrol.core.profiles``."""

from __future__ import annotations

import pytest

from viscontrol.core.profiles import ProductProfile, ProfileStore


def _mkp(name: str = "Default", **overrides) -> ProductProfile:
    data = dict(
        name=name,
        expected_area_px=60_000,
        expected_width_px=290,
        expected_height_px=290,
        fused_threshold=1.4,
        noise_threshold=0.5,
        camera_exposure_us=2000,
        camera_gain=0,
        roi_split_x=1768,
        transfer_line_x=2400,
    )
    data.update(overrides)
    return ProductProfile(**data)


def test_profile_defaults_and_validation() -> None:
    p = _mkp()
    assert p.name == "Default"
    assert p.expected_area_mm2(px_per_mm=10) == pytest.approx(600.0)


def test_profile_name_rejects_path_separators() -> None:
    with pytest.raises(ValueError):
        _mkp(name="evil/profile")
    with pytest.raises(ValueError):
        _mkp(name="evil\\profile")


def test_profile_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        _mkp(fused_threshold=1.0)  # must be > 1
    with pytest.raises(ValueError):
        _mkp(noise_threshold=1.5)  # must be < 1


def test_profile_store_upsert_and_lookup() -> None:
    store = ProfileStore([_mkp("A"), _mkp("B")])
    assert store.names() == ["A", "B"]
    assert "A" in store and "Z" not in store
    store.upsert(_mkp("A", expected_area_px=80_000))  # replace
    assert store.get("A").expected_area_px == 80_000
    store.upsert(_mkp("C"))
    assert store.names() == ["A", "B", "C"]


def test_profile_store_remove_unknown_raises() -> None:
    store = ProfileStore([_mkp("X")])
    with pytest.raises(KeyError):
        store.remove("Y")


def test_profile_store_as_list_is_sorted_by_name() -> None:
    store = ProfileStore([_mkp("Z"), _mkp("A"), _mkp("M")])
    assert [p.name for p in store.as_list()] == ["A", "M", "Z"]
