"""Tests for viscontrol/core/training_data_collector.py.

Pure pytest — no Qt, no camera. All file I/O goes through tmp_path.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from viscontrol.core.training_data_collector import TrainingDataCollector


@dataclass
class _Cfg:
    output_dir: str
    jpeg_quality: int = 88
    normal_sample_every_n_rows: int = 20
    min_free_space_gb: float = 10
    low_disk_warning_cooldown_s: float = 300
    max_total_size_gb: float = 50
    maintenance_interval_minutes: int = 60


class _FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, tuple, dict]] = []

    def _rec(self, level, msg, *a, **kw):
        self.records.append((level, msg, a, kw))

    def debug(self, msg, *a, **kw):
        self._rec("debug", msg, *a, **kw)

    def info(self, msg, *a, **kw):
        self._rec("info", msg, *a, **kw)

    def warning(self, msg, *a, **kw):
        self._rec("warning", msg, *a, **kw)

    def error(self, msg, *a, **kw):
        self._rec("error", msg, *a, **kw)

    def exception(self, msg, *a, **kw):
        self._rec("exception", msg, *a, **kw)

    def count(self, level: str, needle: str) -> int:
        return sum(1 for lvl, msg, _, _ in self.records if lvl == level and needle in msg)


def _make_collector(tmp_path: Path, **overrides) -> tuple[TrainingDataCollector, _FakeLogger, _Cfg]:
    cfg = _Cfg(output_dir=str(tmp_path / "training_data"), **overrides)
    logger = _FakeLogger()
    return TrainingDataCollector(cfg, logger), logger, cfg


def _image(size: int = 32) -> np.ndarray:
    return np.random.randint(0, 255, size=(size, size), dtype=np.uint8)


# ---------- T1 ----------

def test_layout_fault_always_saved(tmp_path: Path) -> None:
    collector, _, _ = _make_collector(tmp_path)
    for frame_id in (1, 7, 100, 101, 999):
        decision = collector.decide(frame_id=frame_id, is_ambiguous=False, layout_fault_active=True)
        assert decision.should_save is True
        assert decision.category == "layout_fault"


# ---------- T2 ----------

def test_ambiguous_always_saved(tmp_path: Path) -> None:
    collector, _, _ = _make_collector(tmp_path)
    decision = collector.decide(frame_id=13, is_ambiguous=True, layout_fault_active=False)
    assert decision.should_save is True
    assert decision.category == "ambiguous"


# ---------- T3 ----------

def test_normal_sampling_rate(tmp_path: Path) -> None:
    collector, _, _ = _make_collector(tmp_path, normal_sample_every_n_rows=20)
    saved_ids = []
    for frame_id in range(1, 101):
        decision = collector.decide(frame_id=frame_id, is_ambiguous=False, layout_fault_active=False)
        if decision.should_save:
            saved_ids.append(frame_id)
    assert saved_ids == [20, 40, 60, 80, 100]
    assert len(saved_ids) == 5


# ---------- T4 ----------

def test_priority_order(tmp_path: Path) -> None:
    collector, _, _ = _make_collector(tmp_path, normal_sample_every_n_rows=20)
    decision = collector.decide(frame_id=40, is_ambiguous=True, layout_fault_active=False)
    assert decision.category == "ambiguous"
    assert decision.should_save is True


# ---------- T5 ----------

def test_disk_guard_blocks_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, logger, cfg = _make_collector(tmp_path, min_free_space_gb=10)

    class _Usage:
        free = 1 * (1024 ** 3)  # 1 GB free, below 10 GB threshold
        total = 100 * (1024 ** 3)
        used = 99 * (1024 ** 3)

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage())

    decision = collector.decide(frame_id=1, is_ambiguous=True, layout_fault_active=False)
    result = collector.save(_image(), decision, frame_id=1)

    assert result is None
    assert logger.count("warning", "SAVE-BLOCKED-LOW-DISK") == 1
    written = list(Path(cfg.output_dir).rglob("*.jpg")) if Path(cfg.output_dir).exists() else []
    assert written == []


# ---------- T6 ----------

def test_disk_guard_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    collector, logger, cfg = _make_collector(
        tmp_path, min_free_space_gb=10, low_disk_warning_cooldown_s=300,
    )

    class _Usage:
        free = 1 * (1024 ** 3)
        total = 100 * (1024 ** 3)
        used = 99 * (1024 ** 3)

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage())
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)

    decision = collector.decide(frame_id=1, is_ambiguous=True, layout_fault_active=False)
    collector.save(_image(), decision, frame_id=1)
    collector.save(_image(), decision, frame_id=2)  # same monotonic time -> within cooldown

    assert logger.count("warning", "SAVE-BLOCKED-LOW-DISK") == 1


# ---------- T7 ----------

def test_jpeg_compression_actually_compresses(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path, jpeg_quality=88)
    image = np.random.randint(0, 255, size=(1000, 1000), dtype=np.uint8)

    decision = collector.decide(frame_id=20, is_ambiguous=False, layout_fault_active=False)
    assert decision.should_save is True
    path = collector.save(image, decision, frame_id=20)

    assert path is not None
    written = Path(path)
    assert written.exists()

    loaded = cv2.imread(str(written), cv2.IMREAD_UNCHANGED)
    assert loaded is not None
    assert loaded.shape[:2] == image.shape[:2]

    uncompressed_size = image.nbytes  # equivalent raw/.npy size
    jpeg_size = written.stat().st_size
    assert jpeg_size < uncompressed_size


# ---------- T8 ----------

def test_cleanup_removes_oldest_first(tmp_path: Path) -> None:
    collector, logger, cfg = _make_collector(tmp_path, max_total_size_gb=1 / 1024)  # 1 MB limit
    out = Path(cfg.output_dir)

    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    chunk = b"x" * (400 * 1024)  # 400 KB per folder
    for d in dates:
        folder = out / "normal_sample" / d
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "a.jpg").write_bytes(chunk)

    collector.run_maintenance()

    remaining = sorted(
        p.name for p in (out / "normal_sample").iterdir() if p.is_dir()
    ) if (out / "normal_sample").exists() else []

    assert "2026-01-01" not in remaining  # oldest removed first
    assert len(remaining) >= 1
    assert len(remaining) < len(dates)  # at least one non-oldest survives
    assert logger.count("info", "CLEANUP-REMOVE") >= 1


# ---------- T9 ----------

def test_cleanup_never_partial_deletes(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path, max_total_size_gb=1 / 1024)
    out = Path(cfg.output_dir)

    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    for d in dates:
        folder = out / "normal_sample" / d
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "a.jpg").write_bytes(b"x" * (400 * 1024))
        (folder / "b.jpg").write_bytes(b"y" * (400 * 1024))

    collector.run_maintenance()

    for d in dates:
        folder = out / "normal_sample" / d
        if folder.exists():
            files = list(folder.iterdir())
            assert len(files) == 2  # fully present, never partially deleted
        # else: fully absent — also fine, just never partial.


# ---------- T10 ----------

def test_disabled_by_default() -> None:
    from viscontrol.core.config import AppConfig

    cfg = AppConfig()
    assert cfg.training_data.enabled is False
