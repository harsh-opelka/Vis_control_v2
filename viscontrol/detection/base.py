"""Common types for the detection module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from viscontrol.core.profiles import ProductProfile


@dataclass
class Detection:
    """One detected blob.

    ``width_px``/``height_px`` are the column-axis and row-axis extents
    respectively (the row axis is configurable via
    ``orientation.row_direction``; the detector receives an already-rotated
    frame so it can just use bbox extents).

    ``label`` ∈ {"single", "row_fused", "column_fused", "unknown"}.
    """

    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixel coords
    centroid: tuple[float, float]
    area_px: int
    width_px: int
    height_px: int
    confidence: float
    label: str
    circularity: float = 0.0  # 4π·area/perimeter²; stored for Learn Reference
    solidity: float = 0.0     # area/convex_hull_area; stored for Learn Reference

    @property
    def is_fault(self) -> bool:
        return self.label in ("row_fused", "unknown")


class Detector(Protocol):
    """Minimal detector interface; lets us swap classical for ML later."""

    def detect(self, image: np.ndarray, profile: ProductProfile) -> list[Detection]:  # noqa: D401
        """Return detections in ``image``'s coordinate space (full ROI)."""
        ...
