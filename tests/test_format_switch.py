"""Tests for PLC-driven product/cloth switching (core/format_switch.py)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from viscontrol.core.config import AppConfig, load_config
from viscontrol.core.format_switch import FormatSwitcher, match_cloth, match_product

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

_PRODUCTS = {
    "Berliner": {"expected_width_px": 312, "expected_height_px": 312, "fused_threshold": 1.4},
    "Apfelschnecken": {"expected_width_px": 318, "expected_height_px": 318, "fused_threshold": 1.8},
}
_CLOTHS = {
    "white": {"dough_is_darker": True, "cloth_brightness_threshold": 112},
    "black": {"dough_is_darker": False, "cloth_brightness_threshold": 48},
}


def _cfg() -> AppConfig:
    cfg = AppConfig.model_validate({
        "profiles": [{"name": "Default"}],
        "products": _PRODUCTS,
        "cloths": _CLOTHS,
        "app": {"active_product": "Berliner", "active_cloth": "white"},
    })
    return cfg


class _Harness:
    def __init__(self) -> None:
        self.t = 0.0
        self.idle = True
        self.applied: list[tuple[str, str]] = []
        self.cfg = _cfg()
        self.sw = FormatSwitcher(
            cfg_fn=lambda: self.cfg,
            apply_fn=lambda p, c: self.applied.append((p, c)),
            is_idle_fn=lambda: self.idle,
            now_fn=lambda: self.t,
        )
        self.sw.apply_defaults()
        self.applied.clear()

    def advance(self, dt: float) -> None:
        self.t += dt
        self.sw.tick()


@pytest.mark.parametrize("raw", ["Apfelschnecken", "  apfelschnecken ", "APFELSCHNECKEN\n"])
def test_product_name_matching_trim_and_case(raw: str) -> None:
    assert match_product(raw, ["Berliner", "Apfelschnecken"]) == "Apfelschnecken"


@pytest.mark.parametrize("raw", ["", "   ", "Apfelschnecke", "Krapfen", None, 5])
def test_product_name_invalid(raw) -> None:
    assert match_product(raw, ["Berliner", "Apfelschnecken"]) is None


def test_cloth_color_mapping() -> None:
    known = ["white", "black"]
    assert match_cloth(0, known) == "white"
    assert match_cloth(1, known) == "black"
    for bad in (2, 255, -1, None, "1", True):
        assert match_cloth(bad, known) is None


def test_switch_applies_after_stable_period() -> None:
    h = _Harness()
    h.sw.submit("apfelschnecken", 1)
    assert h.applied == []          # not stable yet
    h.advance(0.5)
    assert h.applied == []
    h.advance(0.6)
    assert h.applied == [("Apfelschnecken", "black")]
    assert (h.sw.active_product, h.sw.active_cloth) == ("Apfelschnecken", "black")


def test_flapping_value_restarts_stability_timer() -> None:
    h = _Harness()
    h.sw.submit("Apfelschnecken", 1)
    h.advance(0.8)
    h.sw.submit("Berliner", 1)      # changed again before 1 s
    h.advance(0.8)
    assert h.applied == []
    h.advance(0.3)
    assert h.applied == [("Berliner", "black")]


def test_switch_deferred_while_transfer_active_then_applied_when_idle() -> None:
    h = _Harness()
    h.idle = False
    h.sw.submit("Apfelschnecken", 1)
    h.advance(5.0)
    assert h.applied == []
    assert h.sw.active_product == "Berliner"
    h.idle = True
    h.advance(0.25)
    assert h.applied == [("Apfelschnecken", "black")]


def test_unknown_values_keep_last_settings_and_warn_once() -> None:
    from loguru import logger as _l  # viscontrol logger is loguru-based

    h = _Harness()
    h.sw.submit("Apfelschnecken", 1)
    h.advance(1.1)
    h.applied.clear()

    messages: list[str] = []
    sink = _l.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        for _ in range(5):
            h.sw.submit("Krapfen", 7)   # same bad values every poll
            h.advance(2.0)
        h.sw.submit("", None)            # empty + read error
        h.sw.submit(None, None)
        h.advance(2.0)
    finally:
        _l.remove(sink)

    assert h.applied == []
    assert (h.sw.active_product, h.sw.active_cloth) == ("Apfelschnecken", "black")
    # One warning for each distinct bad (kind, value): Krapfen, 7, '', None(cloth); product None dedupes with ''->no
    product_warnings = [m for m in messages if "invalid PLC product 'Krapfen'" in m]
    cloth_warnings = [m for m in messages if "ClothColor 7" in m]
    assert len(product_warnings) == 1
    assert len(cloth_warnings) == 1


def test_one_invalid_component_keeps_other_applied() -> None:
    h = _Harness()
    h.sw.submit("Apfelschnecken", 9)    # valid product, invalid cloth
    h.advance(1.1)
    assert h.applied == [("Apfelschnecken", "white")]


def test_recovery_after_invalid_value() -> None:
    h = _Harness()
    h.sw.submit(None, None)
    h.advance(2.0)
    h.sw.submit("Apfelschnecken", 1)
    h.advance(1.1)
    assert h.applied == [("Apfelschnecken", "black")]


def test_same_as_active_is_not_reapplied() -> None:
    h = _Harness()
    h.sw.submit("Berliner", 0)
    h.advance(3.0)
    assert h.applied == []


def test_disabled_without_products_and_cloths() -> None:
    applied: list = []
    cfg = AppConfig.model_validate({"profiles": [{"name": "Default"}]})
    sw = FormatSwitcher(lambda: cfg, lambda p, c: applied.append((p, c)), lambda: True, lambda: 0.0)
    sw.apply_defaults()
    sw.submit("Berliner", 0)
    sw.tick()
    assert not sw.enabled and applied == []


def test_apply_failure_keeps_previous_state() -> None:
    h = _Harness()

    def boom(p: str, c: str) -> None:
        raise RuntimeError("x")

    h.sw._apply_fn = boom
    h.sw.submit("Apfelschnecken", 1)
    h.advance(1.1)
    assert h.sw.active_product == "Berliner"


def test_old_style_config_still_loads(tmp_path: Path) -> None:
    """A local.yaml without products/cloths/active_* must load unchanged."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copy(CONFIG_DIR / "default.yaml", cfg_dir / "default.yaml")
    local = yaml.safe_load((CONFIG_DIR / "local.yaml").read_text(encoding="utf-8"))
    for key in ("products", "cloths"):
        local.pop(key, None)
    for key in ("active_product", "active_cloth"):
        local["app"].pop(key, None)
    for key in ("node_ext_product_name", "node_ext_cloth_color"):
        local["plc"].pop(key, None)
    (cfg_dir / "local.yaml").write_text(yaml.safe_dump(local), encoding="utf-8")

    cfg = load_config(cfg_dir)
    assert cfg.products == {} and cfg.cloths == {}
    assert cfg.app.active_product == ""
    assert cfg.active_profile().name == "Default"


def test_repo_local_yaml_has_both_products_and_cloths() -> None:
    cfg = load_config(CONFIG_DIR)
    assert set(cfg.products) == {"Berliner", "Apfelschnecken"}
    assert set(cfg.cloths) == {"white", "black"}
    assert cfg.cloths["black"].cloth_brightness_threshold == 48
    assert cfg.cloths["white"].cloth_brightness_threshold == 112
    assert cfg.plc.node_ext_product_name == "ns=6;s=::AsGlobalPV:gFormatSet.Name"
