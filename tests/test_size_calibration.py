"""Tests for ``viscontrol.core.size_calibration`` (SizeCalibrator).

No Qt. No camera. K1-K4/K7-K9 feed ``CalibrationSample`` objects directly to
:class:`SizeCalibrator`, bypassing image processing entirely, so they target
the math (outlier rejection, percentile/margin derivation, sanity gates),
not OpenCV. K10 (duplicate-circle rejection) exercises the real two-pass
``add_frame`` control flow, with ``cv2.HoughCircles`` itself monkeypatched
to return canned circle arrays — this keeps the two-pass logic under test
deterministic without depending on real Hough behavior on a synthetic image.

Tolerance/layout-gap measurement (formerly K5/K6) has been removed from this
module entirely — see calibration_bugfix_result.txt. Cluster tolerance is a
property of depositor spacing, not piece size, and this module no longer
measures or suggests it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from viscontrol.core.config import load_config, save_config
from viscontrol.core.size_calibration import CalibrationSample, SizeCalibrator

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class RecordingLogger:
    """Captures every structured log line the calibrator emits, so tests can
    assert on prefixes without depending on the real loguru sink. Mirrors
    tests/test_layout_quality.py's RecordingLogger."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def _record(self, level: str, msg: str, *args) -> None:
        text = msg.format(*args) if args else msg
        self.lines.append((level, text))

    def debug(self, msg, *args) -> None: self._record("debug", msg, *args)  # noqa: E704
    def info(self, msg, *args) -> None: self._record("info", msg, *args)  # noqa: E704
    def warning(self, msg, *args) -> None: self._record("warning", msg, *args)  # noqa: E704

    def has(self, needle: str) -> bool:
        return any(needle in text for _, text in self.lines)


def mk_cfg(**overrides) -> SimpleNamespace:
    defaults = dict(
        enabled=True,
        capture_frames=30,
        sweep_min_radius_px=20,
        sweep_max_radius_px=400,
        outlier_reject_frac=0.35,
        radius_margin=0.30,
        min_frames=10,
        min_pieces=4,
        max_radius_cv=0.25,
        max_plausible_pieces_per_frame=40,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def feed_samples(calibrator: SizeCalibrator, frames: list[list[float]]) -> None:
    """Feed pre-built per-frame radius lists directly as CalibrationSample
    objects (bypassing add_frame/cv2 entirely)."""
    for i, radii in enumerate(frames):
        tangent_xs = tuple(float(j * 1000) for j in range(len(radii)))
        center_ys = tuple(0.0 for _ in radii)
        calibrator._samples.append(  # noqa: SLF001 — direct sample injection for tests
            CalibrationSample(
                frame_id=i, radii=tuple(float(r) for r in radii),
                tangent_xs=tangent_xs, center_ys=center_ys,
            )
        )


def jitter_radii(base: float, n: int, spread: float) -> list[float]:
    """Deterministic +/- jitter around ``base`` — no RNG so results are
    reproducible without seeding."""
    pattern = [0.0, spread, -spread, spread * 0.5, -spread * 0.5, spread * 0.25]
    return [base + pattern[i % len(pattern)] for i in range(n)]


# ---------------------------------------------------------------------------
# K1 — median radius correct
# ---------------------------------------------------------------------------


def test_k1_median_radius_correct():
    cal = SizeCalibrator(mk_cfg())
    frames = [jitter_radii(145.0, 8, 8.0) for _ in range(30)]
    feed_samples(cal, frames)

    result = cal.compute()

    assert abs(result.radius_median - 145.0) <= 3.0
    assert result.suggested_min_radius_px <= 145 - int(0.30 * 145) + 3
    assert result.suggested_max_radius_px >= 145 + int(0.30 * 145) - 3
    assert result.suggested_min_radius_px < 145.0 < result.suggested_max_radius_px


# ---------------------------------------------------------------------------
# K2 — outlier rejection
# ---------------------------------------------------------------------------


def test_k2_outlier_rejection():
    cal = SizeCalibrator(mk_cfg())
    frames = [jitter_radii(145.0, 8, 8.0) for _ in range(30)]
    # Inject 3 spurious reflection-sized radii of 350 into the first 3 frames.
    for i in range(3):
        frames[i].append(350.0)
    feed_samples(cal, frames)

    result = cal.compute()

    assert abs(result.radius_median - 145.0) <= 3.0
    assert result.radius_p90 < 300.0  # the 350s must not survive into p90


# ---------------------------------------------------------------------------
# K3 / K4 — half size / double size (the 0.5x-2x range this feature exists for)
# ---------------------------------------------------------------------------


def test_k3_half_size():
    cal = SizeCalibrator(mk_cfg())
    frames = [jitter_radii(72.0, 8, 4.0) for _ in range(30)]
    feed_samples(cal, frames)

    result = cal.compute()

    assert abs(result.suggested_min_radius_px - 50) <= 3
    assert abs(result.suggested_max_radius_px - 94) <= 3
    assert abs(result.suggested_expected_width_px - 144) <= 4


def test_k4_double_size():
    cal = SizeCalibrator(mk_cfg())
    frames = [jitter_radii(290.0, 8, 10.0) for _ in range(30)]
    feed_samples(cal, frames)

    result = cal.compute()

    assert abs(result.suggested_min_radius_px - 203) <= 4
    assert abs(result.suggested_max_radius_px - 377) <= 4
    assert abs(result.suggested_expected_width_px - 580) <= 4


# ---------------------------------------------------------------------------
# K7 — too few pieces
# ---------------------------------------------------------------------------


def test_k7_too_few_pieces():
    cal = SizeCalibrator(mk_cfg())
    frames = [jitter_radii(145.0, 2, 3.0) for _ in range(15)]
    feed_samples(cal, frames)

    result = cal.compute()

    assert result.ok is False
    assert result.reason == "too_few_pieces"


# ---------------------------------------------------------------------------
# K8 — too variable radius (mixed sizes on one cloth)
# ---------------------------------------------------------------------------


def test_k8_too_variable_radius():
    cal = SizeCalibrator(mk_cfg())
    # Spread widely enough that even after +/-35% outlier rejection around
    # the median, the survivors still have stddev/median > max_radius_cv.
    base_frame = [80.0, 90.0, 100.0, 110.0, 170.0, 210.0, 230.0, 250.0]
    frames = [list(base_frame) for _ in range(15)]
    feed_samples(cal, frames)

    result = cal.compute()

    assert result.ok is False
    assert result.reason == "too_variable"
    assert result.radius_stddev / result.radius_median > 0.25


# ---------------------------------------------------------------------------
# K9 — no config written without Apply (SizeCalibrator itself is side-effect free)
# ---------------------------------------------------------------------------


DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture()
def fresh_config_dir(tmp_path: Path) -> Path:
    out = tmp_path / "config"
    out.mkdir()
    shutil.copy(DEFAULT_YAML, out / "default.yaml")
    return out


def test_k9_no_config_written_without_apply(fresh_config_dir: Path):
    cfg = load_config(fresh_config_dir)
    save_config(cfg, fresh_config_dir)  # seed a local.yaml to compare against
    local_path = fresh_config_dir / "local.yaml"
    before = local_path.read_bytes()

    cal = SizeCalibrator(cfg.size_calibration)
    frames = [jitter_radii(145.0, 8, 8.0) for _ in range(30)]
    feed_samples(cal, frames)
    result = cal.compute()
    assert result.ok is True  # compute() ran and produced a real result...

    after = local_path.read_bytes()
    assert before == after  # ...but local.yaml is untouched — no side effects.


# ---------------------------------------------------------------------------
# K10 — duplicate-circle rejection (the real "853 pieces per frame" bug)
# ---------------------------------------------------------------------------


def _fake_circles(entries: list[tuple[float, float, float]]) -> np.ndarray | None:
    """Build a cv2.HoughCircles-shaped return value: shape (1, N, 3)."""
    if not entries:
        return None
    return np.array([entries], dtype=np.float32)


def test_k10_duplicate_circle_rejection_cleaned_up(monkeypatch):
    """Pass 1 (loose minDist) returns 400 duplicate detections clustered
    around 20 true piece centers (radius ~174px, reproducing the real
    "853 pieces per frame" bug). Pass 2, re-run with
    minDist = rough_median_radius * 1.3, must collapse this down to a
    plausible per-frame count."""
    true_centers = [(200.0 + i * 400.0, 300.0, 174.0) for i in range(20)]

    pass1_entries: list[tuple[float, float, float]] = []
    for cx, cy, r in true_centers:
        # 20 near-duplicate circles per true piece -> 400 total, exactly
        # the kind of concentric/near-concentric flood a too-small minDist
        # produces on a real ~174px-radius piece.
        for k in range(20):
            pass1_entries.append((cx + k * 0.5, cy + k * 0.3, r + k * 0.2))

    calls = {"n": 0}

    def fake_hough_circles(_image, _method, *, dp, minDist, param1, param2, minRadius, maxRadius):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_circles(pass1_entries)
        # Pass 2: simulate minDist actually being honored — one circle
        # survives per true center.
        assert minDist > 100.0  # scaled to ~174*1.3, not the old fixed 30px
        return _fake_circles(true_centers)

    monkeypatch.setattr(
        "viscontrol.core.size_calibration.cv2.HoughCircles", fake_hough_circles
    )

    logger = RecordingLogger()
    cal = SizeCalibrator(mk_cfg(), logger)
    gray = np.zeros((600, 8400), dtype=np.uint8)

    used = cal.add_frame(gray, 0)

    assert used is True
    assert len(cal._samples) == 1  # noqa: SLF001
    assert len(cal._samples[0].radii) == 20  # noqa: SLF001 — cleaned up, not 400
    assert logger.has("CAL-SWEEP-PASS1")
    assert logger.has("raw_detections=400")
    assert logger.has("CAL-SWEEP-PASS2")
    assert logger.has("final_detections=20")


def test_k10_duplicate_circle_rejection_frame_discarded(monkeypatch):
    """If pass 2 STILL returns an implausible count (minDist scaling wasn't
    enough to clean it up), the whole frame must be discarded — not allowed
    to corrupt the radius median — and logged as CAL-FRAME-REJECTED."""
    pass1_entries = [(100.0 + i * 2.0, 100.0, 50.0) for i in range(300)]
    pass2_entries = [(100.0 + i * 2.0, 100.0, 50.0) for i in range(60)]  # still too many

    calls = {"n": 0}

    def fake_hough_circles(_image, _method, *, dp, minDist, param1, param2, minRadius, maxRadius):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_circles(pass1_entries)
        return _fake_circles(pass2_entries)

    monkeypatch.setattr(
        "viscontrol.core.size_calibration.cv2.HoughCircles", fake_hough_circles
    )

    logger = RecordingLogger()
    cal = SizeCalibrator(mk_cfg(max_plausible_pieces_per_frame=40), logger)
    gray = np.zeros((200, 700), dtype=np.uint8)

    used = cal.add_frame(gray, 7)

    assert used is False
    assert len(cal._samples) == 0  # noqa: SLF001 — frame not stored
    assert logger.has("CAL-FRAME-REJECTED")
    assert logger.has("frame=7")
    assert logger.has("count=60")
