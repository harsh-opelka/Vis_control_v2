"""Calibration / "Learn Reference" support.

Computes the median area / width / height plus mean/min shape descriptors
(circularity and solidity) from an already-detected piece list.  The wizard
(step 7) or the SERVICE view button triggers ``MainWindow._on_learn_reference``,
which runs the CURRENTLY ACTIVE detection method (``detection.method`` — blob /
contour_external / hough / bg_subtract, see ``InspectionPipeline.detect_cloth_pieces``)
and passes the resulting detections in here. :func:`learn_reference` itself
does not run detection and does not care which method produced its input —
every method returns the same ``Detection`` shape (width_px/height_px/area_px/
circularity/solidity/label), so the statistics below work unchanged regardless
of method.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from viscontrol.core.profiles import ProductProfile
from viscontrol.detection.base import Detection


@dataclass
class CalibrationResult:
    """Output of :func:`learn_reference`."""

    profile: ProductProfile
    sample_count: int
    width_px: int
    height_px: int
    area_px: int
    circularity_mean: float
    circularity_min: float
    solidity_mean: float
    solidity_min: float


def learn_reference(
    detections: list[Detection],
    *,
    base_profile: ProductProfile,
    new_name: str | None = None,
    min_samples: int = 3,
) -> CalibrationResult:
    """Store the median geometry + shape of ``detections`` into a (new) profile.

    ``detections`` should be exactly what the active detection method found
    (see ``InspectionPipeline.detect_cloth_pieces`` — the same dispatch used
    by the calibration page's live preview), NOT re-derived here. This keeps
    Learn Reference in lockstep with whichever method is currently active:
    if Hough is active and showing 38 pieces, those are the 38 samples used.

    ``new_name`` may be ``None`` to update ``base_profile`` in place
    (semantically — we return a copy with the new geometry). Otherwise a new
    profile with that name is returned.
    """
    # Only "single" pieces should drive calibration; fused/unknown pollute stats.
    singles = [d for d in detections if d.label == "single"]
    if len(singles) < min_samples:
        raise ValueError(
            f"need at least {min_samples} single pieces to calibrate; got {len(singles)}"
        )
    w = int(round(median(d.width_px for d in singles)))
    h = int(round(median(d.height_px for d in singles)))
    a = int(round(median(d.area_px for d in singles)))

    circ_vals = [d.circularity for d in singles]
    sol_vals = [d.solidity for d in singles]
    circ_mean = sum(circ_vals) / len(circ_vals)
    circ_min = min(circ_vals)
    sol_mean = sum(sol_vals) / len(sol_vals)
    sol_min = min(sol_vals)

    updated = base_profile.model_copy(
        update={
            "name": new_name or base_profile.name,
            "expected_width_px": w,
            "expected_height_px": h,
            "expected_area_px": a,
            "ref_circularity_mean": round(circ_mean, 4),
            "ref_circularity_min": round(circ_min, 4),
            "ref_solidity_mean": round(sol_mean, 4),
            "ref_solidity_min": round(sol_min, 4),
        }
    )
    return CalibrationResult(
        profile=updated,
        sample_count=len(singles),
        width_px=w,
        height_px=h,
        area_px=a,
        circularity_mean=round(circ_mean, 4),
        circularity_min=round(circ_min, 4),
        solidity_mean=round(sol_mean, 4),
        solidity_min=round(sol_min, 4),
    )
