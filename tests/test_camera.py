"""Tests for ``viscontrol.io.camera``.

We exercise:
- :class:`OrientationTransform` for each rotation/flip combo (output shape + a
  marker pixel position so we know rotate-then-flip is applied in the right
  order).
- :class:`MockCamera` reads files from disk and applies the transform.
- :func:`make_camera` falls back to MockCamera when ``"auto"`` and Basler isn't
  available.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from viscontrol.io.camera import MockCamera, OrientationTransform, make_camera


# ---------- OrientationTransform ----------


def _marker_image() -> np.ndarray:
    """A 20×10 image (h=10, w=20) with a single bright pixel at (row 0, col 0).

    After 90° clockwise rotation that pixel should be at (row 0, col 9) of a
    10×10... actually a w=10/h=20 image. We use this to confirm rotation
    direction matches OpenCV's documented behavior.
    """
    img = np.zeros((10, 20), dtype=np.uint8)
    img[0, 0] = 255
    return img


def test_orientation_rotate_0_noop() -> None:
    t = OrientationTransform(rotation=0)
    img = _marker_image()
    out = t.apply(img)
    assert out.shape == img.shape
    assert out[0, 0] == 255


def test_orientation_rotate_90() -> None:
    t = OrientationTransform(rotation=90)
    img = _marker_image()
    out = t.apply(img)
    # 10×20 rotated 90° CW -> 20×10. Top-left becomes top-right.
    assert out.shape == (20, 10)
    assert out[0, -1] == 255


def test_orientation_rotate_180() -> None:
    t = OrientationTransform(rotation=180)
    img = _marker_image()
    out = t.apply(img)
    assert out.shape == img.shape
    assert out[-1, -1] == 255


def test_orientation_rotate_270() -> None:
    t = OrientationTransform(rotation=270)
    img = _marker_image()
    out = t.apply(img)
    assert out.shape == (20, 10)
    assert out[-1, 0] == 255


def test_orientation_flip_only() -> None:
    t = OrientationTransform(rotation=0, flip_horizontal=True)
    img = _marker_image()
    out = t.apply(img)
    assert out.shape == img.shape
    assert out[0, -1] == 255


def test_orientation_rotate_then_flip_order() -> None:
    """rotate 180 + flip_horizontal: pixel at (0,0) -> (h-1,w-1) -> (h-1,0)."""
    t = OrientationTransform(rotation=180, flip_horizontal=True)
    img = _marker_image()
    out = t.apply(img)
    assert out[-1, 0] == 255


# ---------- MockCamera ----------


def _write_image(path: Path, value: int) -> None:
    img = np.full((50, 50), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_mock_camera_cycles_through_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", 10)
    _write_image(tmp_path / "b.png", 20)
    cam = MockCamera(tmp_path, fps=100)
    frame1 = cam.grab()
    frame2 = cam.grab()
    frame3 = cam.grab()
    assert frame1 is not None and frame2 is not None and frame3 is not None
    # Files are sorted, so order is a.png then b.png then back to a.png.
    assert int(frame1[0, 0]) == 10
    assert int(frame2[0, 0]) == 20
    assert int(frame3[0, 0]) == 10


def test_mock_camera_applies_transform(tmp_path: Path) -> None:
    img = np.zeros((10, 20), dtype=np.uint8)
    img[0, 0] = 255
    cv2.imwrite(str(tmp_path / "m.png"), img)
    cam = MockCamera(tmp_path, fps=100, transform=OrientationTransform(rotation=180))
    out = cam.grab()
    assert out is not None
    assert out[-1, -1] == 255


def test_mock_camera_synthetic_when_dir_empty(tmp_path: Path) -> None:
    cam = MockCamera(tmp_path, fps=100, synthetic_size=(64, 64))
    frame = cam.grab()
    assert frame is not None
    assert frame.shape == (64, 64)
    assert frame.max() > 0  # contains the synthetic rectangle / text


def test_mock_camera_start_stop_invokes_callback(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", 30)
    cam = MockCamera(tmp_path, fps=50)
    received: list[np.ndarray] = []
    done = threading.Event()

    def cb(f: np.ndarray) -> None:
        received.append(f)
        if len(received) >= 3:
            done.set()

    cam.start(cb)
    assert done.wait(timeout=2.0), "expected at least 3 frames"
    cam.stop()
    assert len(received) >= 3


def test_mock_camera_double_start_raises(tmp_path: Path) -> None:
    cam = MockCamera(tmp_path, fps=10)
    cam.start(lambda _f: None)
    try:
        with pytest.raises(RuntimeError):
            cam.start(lambda _f: None)
    finally:
        cam.stop()


# ---------- make_camera ----------


def test_make_camera_mock_explicit(tmp_path: Path) -> None:
    cam = make_camera(source="mock", mock_image_dir=tmp_path, mock_fps=5)
    assert isinstance(cam, MockCamera)


def test_make_camera_auto_falls_back_to_mock_when_basler_unavailable(tmp_path: Path) -> None:
    warnings: list[str] = []
    cam = make_camera(
        source="auto",
        mock_image_dir=tmp_path,
        mock_fps=5,
        on_warning=warnings.append,
    )
    # No Basler on the test machine — we should get a MockCamera and a warning.
    assert isinstance(cam, MockCamera)
    # The warning is emitted only when pypylon is missing or no device is found;
    # in either case auto-fallback should leave a string for the status bar.
    assert warnings, "expected at least one fallback warning"


def test_make_camera_basler_explicit_raises_when_unavailable(tmp_path: Path) -> None:
    # pypylon is not installed on the CI box; explicit "basler" must raise.
    with pytest.raises(RuntimeError):
        make_camera(source="basler", mock_image_dir=tmp_path, mock_fps=5)
