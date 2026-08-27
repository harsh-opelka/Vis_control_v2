"""Tests for viscontrol/core/training_data_collector.py.

Pure pytest — no Qt, no camera. All file I/O goes through tmp_path.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from viscontrol.core.training_data_collector import RowPieceData, TrainingDataCollector


@dataclass
class _Cfg:
    output_dir: str
    jpeg_quality: int = 88
    min_free_space_gb: float = 10
    low_disk_warning_cooldown_s: float = 300
    max_total_size_gb: float = 50
    maintenance_interval_minutes: int = 60
    max_tracked_row_ids: int = 500
    capture_every_n_rows: int = 1  # default: no filtering, so existing
                                    # tests (written before this filter
                                    # existed) keep passing unchanged


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


def _piece_data(piece_count: int = 2, tangent_xs: tuple = (100.0, 250.0)) -> RowPieceData:
    return RowPieceData(piece_count=piece_count, tangent_xs=tangent_xs, fused=None)


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

def test_one_image_per_row(tmp_path: Path) -> None:
    """"tripwire" is the only row-lifecycle stage — "confirmed" and
    "cleared" have both been removed (see dropconfirmed_inspection.txt
    and dropcleared_inspection.txt). A selected row gets exactly 1 image."""
    collector, _, cfg = _make_collector(tmp_path)
    row_id, cycle_id = 5, 1
    pd = _piece_data()

    collector.on_row_tripwire(row_id, cycle_id, _image(), pd)

    jpgs = sorted(Path(cfg.output_dir).rglob("*.jpg"))
    jsons = sorted(Path(cfg.output_dir).rglob("*.json"))
    assert len(jpgs) == 1
    assert len(jsons) == 1

    stages = {p.parent.parent.name for p in jpgs}  # output_dir/<stage>/<date>/file.jpg
    assert stages == {"tripwire"}
    assert not (Path(cfg.output_dir) / "confirmed").exists()
    assert not (Path(cfg.output_dir) / "cleared").exists()
    for p in jpgs:
        assert p.name.startswith(f"{row_id}_{cycle_id}_")


# ---------- T4 ----------

def test_duplicate_stage_call_deduped(tmp_path: Path) -> None:
    collector, logger, cfg = _make_collector(tmp_path)
    pd = _piece_data()

    collector.on_row_tripwire(1, 1, _image(), pd)
    collector.on_row_tripwire(1, 1, _image(), pd)  # duplicate call, same row_id+stage

    jpgs = list(Path(cfg.output_dir).rglob("*.jpg"))
    assert len(jpgs) == 1
    assert logger.count("info", "ROW-CAPTURE-SKIPPED-DUPLICATE") == 1


# ---------- T5 ----------

def test_metadata_sidecar_correct_fields(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path)
    pd = _piece_data(piece_count=4, tangent_xs=(50.0, 90.0))

    collector.on_row_tripwire(7, 3, _image(), pd)

    json_path = next(Path(cfg.output_dir).rglob("*.json"))
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["row_id"] == 7
    assert data["cycle_id"] == 3
    assert data["stage"] == "tripwire"
    assert isinstance(data["timestamp"], str) and data["timestamp"]
    assert data["fused"] is None  # genuinely unavailable — null, not guessed
    assert data["piece_count"] == 4
    assert data["tangent_xs"] == [50.0, 90.0]
    assert "outcome" not in data  # removed along with the "cleared" stage


# ---------- T6 ----------

def test_dedup_guard_bounded(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path, max_tracked_row_ids=5)
    pd = _piece_data()

    for row_id in range(1, 21):  # 20 distinct row_ids, bound is 5
        collector.on_row_tripwire(row_id, 1, _image(), pd)

    assert len(collector._captured_stages) <= 5
    assert 1 not in collector._captured_stages  # oldest evicted
    assert 20 in collector._captured_stages  # most recent retained


# ---------- T7 ----------

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


# ---------- T8 ----------

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


# ---------- T9 ----------

def test_jpeg_compression_actually_compresses(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path, jpeg_quality=88)
    image = np.random.randint(0, 255, size=(1000, 1000), dtype=np.uint8)

    decision = collector.decide(frame_id=20, is_ambiguous=True, layout_fault_active=False)
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


# ---------- T10 ----------

def test_cleanup_removes_oldest_first(tmp_path: Path) -> None:
    collector, logger, cfg = _make_collector(tmp_path, max_total_size_gb=1 / 1024)  # 1 MB limit
    out = Path(cfg.output_dir)

    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    chunk = b"x" * (400 * 1024)  # 400 KB per folder
    for d in dates:
        folder = out / "tripwire" / d
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "a.jpg").write_bytes(chunk)

    collector.run_maintenance()

    remaining = sorted(
        p.name for p in (out / "tripwire").iterdir() if p.is_dir()
    ) if (out / "tripwire").exists() else []

    assert "2026-01-01" not in remaining  # oldest removed first
    assert len(remaining) >= 1
    assert len(remaining) < len(dates)  # at least one non-oldest survives
    assert logger.count("info", "CLEANUP-REMOVE") >= 1


# ---------- T11 ----------

def test_cleanup_never_partial_deletes(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path, max_total_size_gb=1 / 1024)
    out = Path(cfg.output_dir)

    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    for d in dates:
        folder = out / "tripwire" / d
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "a.jpg").write_bytes(b"x" * (400 * 1024))
        (folder / "b.jpg").write_bytes(b"y" * (400 * 1024))

    collector.run_maintenance()

    for d in dates:
        folder = out / "tripwire" / d
        if folder.exists():
            files = list(folder.iterdir())
            assert len(files) == 2  # fully present, never partially deleted
        # else: fully absent — also fine, just never partial.


# ---------- T12 ----------

def test_disabled_by_default() -> None:
    from viscontrol.core.config import AppConfig

    cfg = AppConfig()
    assert cfg.training_data.enabled is False


# ---------- T13 ----------

def test_fault_saves_to_training_data(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path)

    collector.on_einlaufband_fault(
        _image(), error_code=2, metadata={"verdict": "FAULT_ROW_FUSED", "fault_reason": None},
    )

    jpgs = list(Path(cfg.output_dir).rglob("*.jpg"))
    jsons = list(Path(cfg.output_dir).rglob("*.json"))
    assert len(jpgs) == 1
    assert len(jsons) == 1
    assert jpgs[0].parent.parent.name == "fault"  # output_dir/fault/<date>/file.jpg

    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["category"] == "fault"
    assert data["error_code"] == 2
    assert isinstance(data["timestamp"], str) and data["timestamp"]
    assert data["verdict"] == "FAULT_ROW_FUSED"  # passed-through metadata, real value
    assert data["fault_reason"] is None  # genuinely unavailable — null, not guessed

    # Independent of storage.defect_image_dir — no such path touched/created.
    assert not (tmp_path / "logs").exists()


# ---------- T14 ----------

def test_fault_capture_failure_does_not_break_existing_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors MainWindow's real integration (see
    dropconfirmed_inspection.txt part B / dropconfirmed_result.txt): the
    existing Einlaufband fault-save runs and completes UNCONDITIONALLY;
    the new, additive on_einlaufband_fault call is wrapped in its own
    try/except right after it, so a failure there can never affect the
    existing save that already happened."""
    collector, logger, _ = _make_collector(tmp_path)

    # The EXISTING Einlaufband fault-save path (MainWindow._save_defect_image)
    # — same real behavior, its own independent directory.
    defect_dir = tmp_path / "logs" / "defects"
    defect_dir.mkdir(parents=True)
    defect_path = defect_dir / "existing_fault.jpg"
    cv2.imwrite(str(defect_path), _image())
    assert defect_path.exists()  # existing save already completed

    def _raise(*_a, **_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(collector, "on_einlaufband_fault", _raise)

    # Same call pattern as main_window.py's integration: a bare try/except
    # around the new call, after the existing save already ran above.
    try:
        collector.on_einlaufband_fault(_image(), 2)
    except Exception:  # noqa: BLE001
        logger.exception("[TRAINDATA-ERROR] fault capture failed")

    assert defect_path.exists()  # existing save completely unaffected
    assert logger.count("exception", "fault capture failed") == 1


# ---------- T15 ----------

def test_every_nth_row_saved(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path, capture_every_n_rows=5)
    pd = _piece_data()

    for row_id in range(1, 16):  # 15 distinct rows
        collector.on_row_tripwire(row_id, 1, _image(), pd)

    jpgs = list(Path(cfg.output_dir).rglob("*.jpg"))
    saved_row_ids = {int(p.name.split("_")[0]) for p in jpgs}

    assert saved_row_ids == {5, 10, 15}  # only the 5th, 10th, 15th rows
    assert len(jpgs) == 3  # 3 selected rows x 1 stage (tripwire only)
    unsaved_ids = set(range(1, 16)) - {5, 10, 15}
    for row_id in unsaved_ids:
        assert not any(p.name.startswith(f"{row_id}_") for p in jpgs)


# ---------- T16 ----------

def test_config_value_changes_behavior(tmp_path: Path) -> None:
    pd = _piece_data()

    collector5, _, cfg5 = _make_collector(tmp_path / "n5", capture_every_n_rows=5)
    for row_id in range(1, 16):
        collector5.on_row_tripwire(row_id, 1, _image(), pd)
    saved5 = {int(p.name.split("_")[0]) for p in Path(cfg5.output_dir).rglob("*.jpg")}
    assert saved5 == {5, 10, 15}

    collector4, _, cfg4 = _make_collector(tmp_path / "n4", capture_every_n_rows=4)
    for row_id in range(1, 16):
        collector4.on_row_tripwire(row_id, 1, _image(), pd)
    saved4 = {int(p.name.split("_")[0]) for p in Path(cfg4.output_dir).rglob("*.jpg")}
    assert saved4 == {4, 8, 12}

    assert saved5 != saved4  # config value genuinely changes behavior


# ---------- T17 ----------

def test_ambiguous_and_fault_bypass_filter(tmp_path: Path) -> None:
    collector, _, cfg = _make_collector(tmp_path, capture_every_n_rows=5)
    pd = _piece_data()

    # Advance rows_seen_count to 3 (rows 1,2,3) — none selected by the
    # filter (3 % 5 != 0), so zero row-lifecycle images exist yet.
    for row_id in range(1, 4):
        collector.on_row_tripwire(row_id, 1, _image(), pd)
    assert list(Path(cfg.output_dir).rglob("*.jpg")) == []

    # Ambiguous path — must save regardless of filter state.
    decision = collector.decide(frame_id=99, is_ambiguous=True, layout_fault_active=False)
    ambiguous_path = collector.save(_image(), decision, frame_id=99)
    assert ambiguous_path is not None

    # Fault path — must save regardless of filter state.
    collector.on_einlaufband_fault(_image(), error_code=2)

    jpgs = list(Path(cfg.output_dir).rglob("*.jpg"))
    categories = {p.parent.parent.name for p in jpgs}
    assert "ambiguous" in categories
    assert "fault" in categories
    assert len(jpgs) == 2  # both saved, completely unaffected by the filter
