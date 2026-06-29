"""Tests for ``viscontrol.core.config``."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from viscontrol.core.config import AppConfig, load_config, save_config
from viscontrol.core.security import verify_pin

# Path to the version-controlled defaults.
DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture()
def fresh_config_dir(tmp_path: Path) -> Path:
    """A temp config dir seeded with the real default.yaml."""
    out = tmp_path / "config"
    out.mkdir()
    shutil.copy(DEFAULT_YAML, out / "default.yaml")
    return out


def test_load_default_config(fresh_config_dir: Path) -> None:
    cfg = load_config(fresh_config_dir)
    assert isinstance(cfg, AppConfig)
    assert cfg.app.mode == "demo"
    assert cfg.app.language == "en"
    assert cfg.app.active_profile == "Default"
    assert cfg.profile_store().has("Default")


def test_load_initializes_pin_hash(fresh_config_dir: Path) -> None:
    cfg = load_config(fresh_config_dir)
    # On first load, the empty hash gets populated from "0000".
    assert cfg.ui.service_pin_hash != ""
    assert verify_pin("0000", cfg.ui.service_pin_hash)


def test_local_overrides_default(fresh_config_dir: Path) -> None:
    (fresh_config_dir / "local.yaml").write_text(
        yaml.safe_dump({"app": {"mode": "production", "language": "de"}}),
        encoding="utf-8",
    )
    cfg = load_config(fresh_config_dir)
    assert cfg.app.mode == "production"
    assert cfg.app.language == "de"
    # Unchanged keys still come from defaults.
    assert cfg.app.active_profile == "Default"


def test_save_round_trip(fresh_config_dir: Path) -> None:
    cfg = load_config(fresh_config_dir)
    cfg.app.mode = "production"
    cfg.app.language = "de"
    save_config(cfg, fresh_config_dir)

    cfg2 = load_config(fresh_config_dir)
    assert cfg2.app.mode == "production"
    assert cfg2.app.language == "de"


def test_active_profile_falls_back_when_missing(fresh_config_dir: Path) -> None:
    cfg = load_config(fresh_config_dir)
    cfg.app.active_profile = "DoesNotExist"
    p = cfg.active_profile()
    assert p.name == "Default"
    assert cfg.app.active_profile == "Default"


def test_load_requires_at_least_one_profile(fresh_config_dir: Path) -> None:
    (fresh_config_dir / "local.yaml").write_text(
        yaml.safe_dump({"profiles": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(fresh_config_dir)


def test_load_missing_defaults_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)
