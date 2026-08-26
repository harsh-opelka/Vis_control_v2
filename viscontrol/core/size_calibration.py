"""Development-only piece-size measurement tool.

Dough piece size can vary roughly 0.5x-2x (a 4x area range) between batches.
Several detection parameters are tuned to the CURRENT piece size and fail
outside it — most critically ``detection.hough.min_radius_px``/
``max_radius_px``: an out-of-band piece produces zero Hough circles for that
piece, not a low-confidence detection (see viscontrol/detection/classical.py
``detect_hough``).

This module measures the actual piece RADIUS from a stationary cloth (a
two-pass wide-open Hough sweep + outlier-robust statistics) and produces
SUGGESTED ``detection.hough.min_radius_px``/``max_radius_px`` values. It
never writes anything itself — see the wizard's "Size Calibration" section
(viscontrol/ui/views/wizard_view.py) for the explicit, operator-confirmed
Apply step that persists the suggestions to config/local.yaml.

This module intentionally does NOT measure or suggest
``proximity_clustering.tolerance_px`` (or anything gap/layout-related).
Cluster tolerance is a property of how the OPERATOR spaces dough on the
cloth, not of piece size — conflating the two was a design mistake in an
earlier revision of this module and has been removed. See
viscontrol/core/layout_quality.py for the (unrelated, unmodified) module
that already owns layout-separability measurement.

This is a SETUP-TIME tool only. It does not touch the production detection
pipeline (viscontrol/detection/pipeline.py), the orchestrator, the clustering
algorithm, or the layout monitor — it is a measurement instrument that runs
alongside them, sharing only the same Hough call shape.

No Qt. No PySide6. No viscontrol.ui imports. cv2 is used directly here (this
module runs its own wide-open Hough sweep) — the only detection module
allowed to do so outside viscontrol/detection.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import cv2
import numpy as np


class SizeCalibrationConfig(Protocol):
    """Shape of the config object this module reads (see
    viscontrol.core.config._SizeCalibrationSection). Duck-typed so tests
    don't need the real pydantic model."""

    enabled: bool
    capture_frames: int
    sweep_min_radius_px: int
    sweep_max_radius_px: int
    outlier_reject_frac: float
    radius_margin: float
    min_frames: int
    min_pieces: int
    max_radius_cv: float
    max_plausible_pieces_per_frame: int


class _LoggerLike(Protocol):
    def debug(self, *args: Any, **kwargs: Any) -> None: ...
    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, *args: Any, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class CalibrationSample:
    """One frame's wide-open Hough results (post two-pass duplicate
    rejection — see :meth:`SizeCalibrator.add_frame`)."""

    frame_id: int
    radii: tuple[float, ...]
    tangent_xs: tuple[float, ...]  # center_x - radius, per circle
    center_ys: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationResult:
    ok: bool
    reason: str  # "ok" | "too_few_frames" | "too_few_pieces" | "too_variable"
    frames_used: int
    pieces_per_frame_median: float
    radius_median: float
    radius_p10: float
    radius_p90: float
    radius_stddev: float
    # derived values, ready to write to config
    suggested_min_radius_px: int
    suggested_max_radius_px: int
    suggested_expected_area_px: int
    suggested_expected_width_px: int
    suggested_expected_height_px: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile, ``pct`` in [0, 100]. Assumes
    ``sorted_values`` is already sorted ascending and non-empty."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


class SizeCalibrator:
    """Accumulates wide-open Hough samples from a stationary cloth and
    derives suggested ``detection.hough`` radius limits.

    Usage: call :meth:`add_frame` once per captured frame (see the wizard's
    Measure flow), then :meth:`compute` once all frames are in. Call
    :meth:`reset` to start a fresh measurement (e.g. "Measure" pressed
    again without leaving the page).
    """

    def __init__(self, cfg: SizeCalibrationConfig, logger: _LoggerLike | None = None) -> None:
        self._cfg = cfg
        self._logger = logger
        self._samples: list[CalibrationSample] = []
        self._started = False

    def reset(self) -> None:
        self._samples = []
        self._started = False

    # ---------- capture ----------

    def add_frame(self, gray_roi: np.ndarray, frame_id: int, *, dp: float = 1.2,
                  param1: float = 80.0, param2: float = 45.0) -> bool:
        """Run a two-pass wide-open Hough sweep on one frame and store the
        sample.

        ``gray_roi`` should be the same crop the production Hough call would
        see (cloth ROI after ``profile.cloth_crop``), single-channel or BGR
        (converted here). ``dp``/``param1``/``param2`` are kept identical to
        the production call (viscontrol/detection/classical.py
        ``detect_hough``) — only the radius bounds and ``minDist`` differ,
        since this measures piece size with the size constraint lifted.

        ``minDist`` is NOT taken from the production config here: a fixed value tied to
        ``sweep_min_radius_px`` (the wide-open search FLOOR, unrelated to
        the actual piece size) massively under-constrains circle separation
        for any real piece bigger than that floor, producing dozens of
        concentric/near-duplicate circles per real piece. Instead:

          Pass 1: loose ``minDist = sweep_min_radius_px`` — accepts
                  duplicates, but gives a ROUGH median radius (the median is
                  robust to duplicate clusters sitting near each true
                  center).
          Pass 2: re-run on the SAME frame with
                  ``minDist = rough_median_radius * 1.3`` — now scaled to
                  the piece size this frame actually has, which suppresses
                  the duplicates. Pass 2's circles become this frame's
                  ``CalibrationSample``.

        If pass 2 still returns more than
        ``cfg.max_plausible_pieces_per_frame`` detections, the frame is
        discarded entirely (not stored, not counted) rather than letting an
        implausible count corrupt the radius median.

        Returns True if the sample was usable and stored, False if the
        image was empty or the frame was rejected as implausible — the
        caller's frame-counter should only advance on a True return if it
        wants exactly ``capture_frames`` usable samples.
        """
        if gray_roi is None or gray_roi.size == 0:
            return False

        if gray_roi.ndim == 3:
            gray = (
                cv2.cvtColor(gray_roi, cv2.COLOR_BGR2GRAY) if gray_roi.shape[2] == 3
                else cv2.cvtColor(gray_roi, cv2.COLOR_BGRA2GRAY)
            )
        else:
            gray = gray_roi

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sweep_min = self._cfg.sweep_min_radius_px
        sweep_max = self._cfg.sweep_max_radius_px

        if self._logger is not None and not self._started:
            self._started = True
            self._logger.info(
                "[CAL] CAL-START frames_requested={} sweep_radius=[{},{}]",
                self._cfg.capture_frames, sweep_min, sweep_max,
            )

        # --- pass 1: loose minDist, rough radius estimate ---
        pass1_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=float(sweep_min),
            param1=param1, param2=param2, minRadius=sweep_min, maxRadius=sweep_max,
        )
        pass1_radii = (
            [float(r) for _cx, _cy, r in pass1_circles[0]] if pass1_circles is not None else []
        )
        rough_median_radius = statistics.median(pass1_radii) if pass1_radii else 0.0

        if self._logger is not None:
            self._logger.debug(
                "[CAL] CAL-SWEEP-PASS1 frame={} raw_detections={} rough_median_radius={:.1f}",
                frame_id, len(pass1_radii), rough_median_radius,
            )

        # --- pass 2: minDist scaled to the piece size pass 1 found ---
        min_dist2 = max(1.0, rough_median_radius * 1.3)
        pass2_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist2,
            param1=param1, param2=param2, minRadius=sweep_min, maxRadius=sweep_max,
        )

        radii: list[float] = []
        tangent_xs: list[float] = []
        center_ys: list[float] = []
        if pass2_circles is not None:
            for cx, cy, r in pass2_circles[0]:
                radii.append(float(r))
                tangent_xs.append(float(cx) - float(r))
                center_ys.append(float(cy))

        if self._logger is not None:
            self._logger.debug(
                "[CAL] CAL-SWEEP-PASS2 frame={} min_dist_used={:.1f} final_detections={}",
                frame_id, min_dist2, len(radii),
            )

        if len(radii) > self._cfg.max_plausible_pieces_per_frame:
            if self._logger is not None:
                self._logger.warning(
                    "[CAL] CAL-FRAME-REJECTED frame={} reason=implausible_count count={}",
                    frame_id, len(radii),
                )
            return False

        sample = CalibrationSample(
            frame_id=frame_id,
            radii=tuple(radii),
            tangent_xs=tuple(tangent_xs),
            center_ys=tuple(center_ys),
        )
        self._samples.append(sample)

        if self._logger is not None:
            self._logger.debug(
                "[CAL] CAL-FRAME frame={} pieces={} radii={}",
                frame_id, len(radii), [round(r, 1) for r in radii],
            )
        return True

    # ---------- computation ----------

    def compute(self) -> CalibrationResult:
        cfg = self._cfg
        warnings: list[str] = []

        frames_used = len(self._samples)
        pieces_counts = [len(s.radii) for s in self._samples]
        pieces_per_frame_median = (
            float(statistics.median(pieces_counts)) if pieces_counts else 0.0
        )

        all_radii = [r for s in self._samples for r in s.radii]

        if not all_radii:
            return self._empty_result(
                frames_used, pieces_per_frame_median,
                reason="too_few_frames" if frames_used < cfg.min_frames else "too_few_pieces",
                warnings=("No circles detected in any captured frame.",),
            )

        # --- outlier rejection (STEP 1b) ---
        all_radii_sorted = sorted(all_radii)
        median_before = statistics.median(all_radii_sorted)
        lo = median_before * (1.0 - cfg.outlier_reject_frac)
        hi = median_before * (1.0 + cfg.outlier_reject_frac)
        survivors = [r for r in all_radii_sorted if lo <= r <= hi]
        discarded = len(all_radii_sorted) - len(survivors)

        if not survivors:
            # Every radius was rejected as an outlier of itself (degenerate);
            # fall back to the unfiltered set rather than crash.
            survivors = all_radii_sorted

        survivors_sorted = sorted(survivors)
        radius_median = statistics.median(survivors_sorted)
        radius_p10 = _percentile(survivors_sorted, 10)
        radius_p90 = _percentile(survivors_sorted, 90)
        radius_stddev = statistics.pstdev(survivors_sorted) if len(survivors_sorted) > 1 else 0.0

        if self._logger is not None:
            self._logger.debug(
                "[CAL] CAL-OUTLIER-REJECT total={} kept={} discarded={} median_before={:.1f} median_after={:.1f}",
                len(all_radii_sorted), len(survivors_sorted), discarded, median_before, radius_median,
            )

        radius_cv = (radius_stddev / radius_median) if radius_median > 0 else float("inf")

        # --- STEP 1c: derived detection parameters ---
        r = radius_median
        suggested_min_radius_px = round(r * (1.0 - cfg.radius_margin))
        suggested_max_radius_px = round(r * (1.0 + cfg.radius_margin))
        suggested_expected_area_px = round(math.pi * r * r)
        suggested_expected_width_px = round(2 * r)
        suggested_expected_height_px = round(2 * r)

        # --- sanity gates (radius-only; no layout/tolerance measurement) ---
        ok = True
        reason = "ok"
        if frames_used < cfg.min_frames:
            ok = False
            reason = "too_few_frames"
        elif pieces_per_frame_median < cfg.min_pieces:
            ok = False
            reason = "too_few_pieces"
        elif radius_cv > cfg.max_radius_cv:
            ok = False
            reason = "too_variable"

        if self._logger is not None:
            self._logger.info(
                "[CAL] CAL-RESULT ok={} reason={} frames={} pieces_median={:.1f} "
                "radius_median={:.1f} radius_p10={:.1f} radius_p90={:.1f} cv={:.3f}",
                ok, reason, frames_used, pieces_per_frame_median,
                radius_median, radius_p10, radius_p90, radius_cv,
            )
            self._logger.info(
                "[CAL] CAL-SUGGEST min_radius={} max_radius={} area={} w={} h={}",
                suggested_min_radius_px, suggested_max_radius_px,
                suggested_expected_area_px, suggested_expected_width_px,
                suggested_expected_height_px,
            )
            for w in warnings:
                self._logger.warning("[CAL] CAL-WARNING {}", w)
            if not ok:
                self._logger.warning("[CAL] CAL-ABORTED reason={}", reason)

        return CalibrationResult(
            ok=ok,
            reason=reason,
            frames_used=frames_used,
            pieces_per_frame_median=pieces_per_frame_median,
            radius_median=radius_median,
            radius_p10=radius_p10,
            radius_p90=radius_p90,
            radius_stddev=radius_stddev,
            suggested_min_radius_px=suggested_min_radius_px,
            suggested_max_radius_px=suggested_max_radius_px,
            suggested_expected_area_px=suggested_expected_area_px,
            suggested_expected_width_px=suggested_expected_width_px,
            suggested_expected_height_px=suggested_expected_height_px,
            warnings=tuple(warnings),
        )

    # ---------- internals ----------

    def _empty_result(
        self, frames_used: int, pieces_per_frame_median: float, *, reason: str,
        warnings: tuple[str, ...],
    ) -> CalibrationResult:
        if self._logger is not None:
            self._logger.info(
                "[CAL] CAL-RESULT ok={} reason={} frames={} pieces_median={:.1f} "
                "radius_median={:.1f} radius_p10={:.1f} radius_p90={:.1f} cv={:.3f}",
                False, reason, frames_used, pieces_per_frame_median, 0.0, 0.0, 0.0, 0.0,
            )
            self._logger.warning("[CAL] CAL-ABORTED reason={}", reason)
        return CalibrationResult(
            ok=False,
            reason=reason,
            frames_used=frames_used,
            pieces_per_frame_median=pieces_per_frame_median,
            radius_median=0.0,
            radius_p10=0.0,
            radius_p90=0.0,
            radius_stddev=0.0,
            suggested_min_radius_px=0,
            suggested_max_radius_px=0,
            suggested_expected_area_px=0,
            suggested_expected_width_px=0,
            suggested_expected_height_px=0,
            warnings=warnings,
        )
