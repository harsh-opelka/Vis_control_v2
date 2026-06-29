"""Tests for the detection pipeline.

Uses synthetic images: bright filled circles (single dough pieces) on a dark
background, drawn directly on numpy arrays. We deliberately use fairly small
canvases (so tests are fast) but scale ``ProductProfile`` to match.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from viscontrol.core.events import Verdict
from viscontrol.core.profiles import ProductProfile
from viscontrol.detection.calibration import learn_reference
from viscontrol.detection.classical import ClassicalDetector
from viscontrol.detection.pipeline import InspectionPipeline


def _mk_profile(**kw) -> ProductProfile:
    base = dict(
        name="Test",
        expected_area_px=2500,   # π * 28² ≈ 2463 — close enough
        expected_width_px=56,    # 2 * radius 28
        expected_height_px=56,
        fused_threshold=1.4,
        noise_threshold=0.3,
        camera_exposure_us=1000,
        camera_gain=0,
        roi_split_x=400,         # full-frame width 800: belt is x<400, cloth is x>=400
        transfer_line_x=700,     # 300px into the cloth ROI
        # Synthetic test images use bright pieces on a dark background — opposite of
        # the real-world dark-dough-on-bright-cloth default.
        dough_is_darker=False,
    )
    base.update(kw)
    return ProductProfile(**base)


def _blank(h: int = 300, w: int = 800) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _draw_piece(img: np.ndarray, center: tuple[int, int], radius: int = 28) -> None:
    cv2.circle(img, center, radius, 255, thickness=-1)


def _draw_row(
    img: np.ndarray, x: int, y_centers: list[int], radius: int = 28
) -> None:
    for y in y_centers:
        _draw_piece(img, (x, y), radius)


# ---------- ClassicalDetector ----------


def test_classical_detects_single_pieces() -> None:
    detector = ClassicalDetector()
    profile = _mk_profile()
    img = _blank()
    # Three well-separated singles in the belt ROI.
    for x in (60, 160, 260):
        _draw_piece(img, (x, 150))

    detections = detector.detect(img[:, :400], profile)
    assert len(detections) == 3
    assert all(d.label == "single" for d in detections)


def test_classical_detects_row_fused() -> None:
    """A blob much wider than expected = two donuts stuck horizontally = row_fused (FAULT)."""
    detector = ClassicalDetector()
    profile = _mk_profile()
    img = _blank()
    # Horizontal pill ~ 120 wide, ~ 50 tall.  Width ratio = 120/56 ≈ 2.14 > 1.4.
    cv2.ellipse(img, (200, 150), (60, 25), 0, 0, 360, 255, -1)

    detections = detector.detect(img[:, :400], profile)
    labels = [d.label for d in detections]
    assert "row_fused" in labels


def test_classical_detects_column_fused() -> None:
    """A blob much taller than expected = two donuts stacked vertically = column_fused (INFO)."""
    detector = ClassicalDetector()
    profile = _mk_profile()
    img = _blank()
    # Vertical pill ~ 50 wide, ~ 120 tall.
    cv2.ellipse(img, (200, 150), (25, 60), 0, 0, 360, 255, -1)
    detections = detector.detect(img[:, :400], profile)
    labels = [d.label for d in detections]
    assert "column_fused" in labels


def test_classical_filters_noise_below_threshold() -> None:
    detector = ClassicalDetector()
    profile = _mk_profile()
    img = _blank()
    _draw_piece(img, (200, 150), radius=28)  # real piece
    _draw_piece(img, (60, 60), radius=4)     # tiny speck below noise threshold
    detections = detector.detect(img[:, :400], profile)
    assert len(detections) == 1


def test_classical_returns_empty_on_blank_image() -> None:
    detector = ClassicalDetector()
    profile = _mk_profile()
    assert detector.detect(_blank()[:, :400], profile) == []


def test_classical_invalid_blur_ksize() -> None:
    with pytest.raises(ValueError):
        ClassicalDetector(blur_ksize=4)  # even
    with pytest.raises(ValueError):
        ClassicalDetector(blur_ksize=0)


# ---------- InspectionPipeline ----------


def test_pipeline_split_rois() -> None:
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile()
    img = _blank()
    belt, cloth = pipe.split_rois(img, profile)
    assert belt.shape == (300, 400)
    assert cloth.shape == (300, 400)


def test_pipeline_clean_belt_emits_ok() -> None:
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile()
    img = _blank()
    # A clean row of 3 singles near the right edge of the belt ROI.
    _draw_row(img, x=350, y_centers=[80, 150, 220])
    result = pipe.run_belt_inspection(img, profile)
    assert result.verdict == Verdict.OK


def test_pipeline_row_fused_emits_fault() -> None:
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile()
    img = _blank()
    # One normal single + one wide (horizontally stuck) blob at x=350.
    _draw_piece(img, (350, 80))
    cv2.ellipse(img, (350, 200), (60, 25), 0, 0, 360, 255, -1)
    result = pipe.run_belt_inspection(img, profile)
    assert result.verdict == Verdict.FAULT_ROW_FUSED


def test_pipeline_column_fused_emits_info() -> None:
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile()
    img = _blank()
    # Row at x=350 with a tall (vertically stacked) blob — column_fused, informational.
    cv2.ellipse(img, (350, 100), (25, 60), 0, 0, 360, 255, -1)
    _draw_piece(img, (350, 220))
    result = pipe.run_belt_inspection(img, profile)
    assert result.verdict == Verdict.INFO_COLUMN_FUSED


def test_pipeline_empty_belt_is_incomplete_row() -> None:
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile()
    img = _blank()
    result = pipe.run_belt_inspection(img, profile)
    assert result.verdict == Verdict.INFO_INCOMPLETE_ROW


def test_pipeline_cloth_tracking_crossing_moved_to_mainwindow() -> None:
    # Transfer-line crossing detection was moved from the stateless pipeline to
    # MainWindow, which holds the per-session baseline.  The pipeline always
    # returns crossed_transfer_line=False; MainWindow computes crossings by
    # comparing n_past_line against a session baseline.
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile()  # transfer_line_x = 700, roi_split_x = 400 → line_local = 300
    img = _blank()
    # Piece at cloth-local x=100 has crossed the line (100 < 300 = line_local).
    _draw_piece(img, (500, 150))  # cloth-local x = 100, past the line
    # Piece at cloth-local x=320 is behind the line (320 > 300).
    _draw_piece(img, (720, 150))  # cloth-local x = 320, behind the line
    result = pipe.run_cloth_tracking(img, profile)
    # Pipeline is now stateless: crossing always False.
    assert result.crossed_transfer_line is False
    # But detections are still returned with correct coords.
    assert len(result.detections) == 2


def test_pipeline_cloth_tracking_no_crossing() -> None:
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile()
    img = _blank()
    _draw_piece(img, (500, 150))  # only piece, well before the line
    result = pipe.run_cloth_tracking(img, profile)
    assert result.crossed_transfer_line is False


def test_pipeline_cloth_tracking_handles_misconfigured_transfer_line() -> None:
    pipe = InspectionPipeline(ClassicalDetector())
    profile = _mk_profile(transfer_line_x=10_000)  # way outside the cloth ROI
    img = _blank()
    _draw_piece(img, (500, 150))
    result = pipe.run_cloth_tracking(img, profile)
    assert result.crossed_transfer_line is False


# ---------- Calibration ----------


def test_learn_reference_median_geometry() -> None:
    detector = ClassicalDetector()
    profile = _mk_profile(expected_width_px=200, expected_height_px=200, expected_area_px=20_000)
    # Pre-calibration profile: expected dimensions are much larger than the test
    # circles (radius 28 → 56 px wide vs expected 200 px).  Use permissive
    # single_min_ratio and fused_threshold so these small circles are classified
    # as "single" rather than "unknown" or "fused".
    profile = profile.model_copy(update={
        "noise_threshold": 0.01,
        "fused_threshold": 5.0,
        "single_min_ratio": 0.1,   # allow blobs as small as 10% of expected
    })

    img = _blank(400, 600)
    for x in (100, 200, 300, 400):
        _draw_piece(img, (x, 200), radius=28)
    result = learn_reference(detector, img, base_profile=profile, new_name="Learned")

    assert result.sample_count == 4
    assert result.profile.name == "Learned"
    # Median diameter of a radius-28 circle is ~ 56.
    assert 50 <= result.width_px <= 60
    assert 50 <= result.height_px <= 60


def test_learn_reference_requires_minimum_samples() -> None:
    detector = ClassicalDetector()
    profile = _mk_profile().model_copy(update={"noise_threshold": 0.01, "fused_threshold": 5.0})
    img = _blank()
    _draw_piece(img, (200, 150), radius=28)  # only one piece
    with pytest.raises(ValueError):
        learn_reference(detector, img, base_profile=profile, min_samples=3)
