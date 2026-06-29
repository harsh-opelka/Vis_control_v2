"""Top-level detection pipeline used by the inference thread.

Splits each frame into BELT (left of ``profile.roi_split_x``) and CLOTH (right
of it). An optional per-ROI ``CropRegion`` (from the profile) is then applied
before detection so machine-frame dark areas are excluded.

- ``TRACKING``: detect on CLOTH ROI (after cloth_crop); if any front-row
  centroid crossed the ``transfer_line_x`` boundary, return a trigger hint.
- ``INSPECTING``: detect on BELT ROI (after belt_crop); find the newest
  (rightmost) row, classify each blob, and emit a verdict.

Coordinate convention
---------------------
Detection centroids and bboxes returned by this module are always in the
coordinate space of the **full ROI slice** (belt or cloth), NOT in the smaller
crop-local sub-image.  The crop offset is added inside ``run_*`` so callers
never need to correct for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from viscontrol.core.events import PipelineResult, Verdict
from viscontrol.core.logger import logger
from viscontrol.core.profiles import CropRegion, ProductProfile
from viscontrol.detection.base import Detection, Detector
from viscontrol.detection.classical import ClassicalDetector as _ClassicalDetector
from viscontrol.detection.row_grouping import median_piece_diameter

if TYPE_CHECKING:
    from viscontrol.core.config import _DetectionSection as DetectionSettings


# ---------------------------------------------------------------------------
# DIAGNOSTIC: two-rows-at-once investigation (see InspectionPipeline.
# analyze_row_profile and ClothTrackingResult.row_profile below). Set to
# False to disable the profile computation/overlay/logging completely with
# zero effect on detection, tripwire, or the state machine. Safe to delete
# this whole block (and everything tagged "DIAGNOSTIC" below) once done.
# ---------------------------------------------------------------------------
DIAGNOSTIC_ROW_PROFILE = False


# ---------------------------------------------------------------------------
# EXPERIMENTAL: ring/hollow-dough detection (cloth-only — belt detection and
# the tripwire's raw occupancy binary are never affected by this flag).
#
# A genuinely solid Berliner can threshold as a hollow RING when a specular
# highlight on its dome punches a fake "hole" through to background. The
# learned-reference solidity gate then rejects it as debris. ``fill_mask_holes``
# is now config-driven (detection.fill_mask_holes / detection.fill_mask_holes_kernel,
# see core/config.py) rather than hardcoded here, but only applies when
# detection.method="blob" — see InspectionPipeline.run_cloth_tracking.
# RELAX_SOLIDITY toggle to False to fall back to the exact previous behavior
# for comparison.
# ---------------------------------------------------------------------------
RELAX_SOLIDITY = False
RELAXED_MIN_SOLIDITY_SHAPE = 0.35   # floor applied instead of the profile's
                                     # ref_solidity_min * solidity_tolerance,
                                     # only when that gate would otherwise be
                                     # stricter. Circularity gate is untouched.


@dataclass
class ClothTrackingResult:
    """Outcome of one CLOTH-ROI pass while TRACKING."""

    front_row_centroids: list[tuple[float, float]]
    crossed_transfer_line: bool
    detections: list[Detection]
    inference_ms: float
    crop_rect: Optional[tuple] = None  # (x1, y1, x2, y2) in cloth-ROI coords
    tripwire_occupancy: float = 0.0    # fraction of strip pixels classified as dough
    tripwire_occupied: bool = False    # True when occupancy >= tripwire_occupancy_threshold
    # DIAGNOSTIC ONLY — not read by tripwire/detection/state-machine code.
    # Column-sum of the existing cloth binary mask along the travel direction
    # (one value per column = dough-pixel count in that column), in
    # cloth-crop-local coordinates. None when DIAGNOSTIC_ROW_PROFILE is False
    # or no binary mask was available this frame.
    row_profile: Optional[np.ndarray] = None
    row_profile_scale: float = 1.0  # cloth-crop-local x * scale = row_profile index


class InspectionPipeline:
    """Stateless pipeline driven by the detection thread."""

    def __init__(self, detector: Detector) -> None:
        self._detector = detector
        # Throttle for the "which method / how many pieces" cloth-detection
        # summary log — once every couple seconds, not per-frame.
        self._last_method_log_time: float = 0.0

    # ---------- ROI helpers ----------

    @staticmethod
    def split_rois(
        frame: np.ndarray, profile: ProductProfile
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (belt_roi, cloth_roi)."""
        if frame.ndim not in (2, 3):
            raise ValueError(f"unexpected frame ndim: {frame.ndim}")
        split = max(1, min(profile.roi_split_x, frame.shape[1] - 1))
        if split != profile.roi_split_x:
            logger.warning(
                "roi_split_x={} out of bounds for frame width={}; clamped to {}",
                profile.roi_split_x, frame.shape[1], split,
            )
        belt = frame[:, :split]
        cloth = frame[:, split:]
        return belt, cloth

    @staticmethod
    def apply_crop(
        roi: np.ndarray, crop: CropRegion
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Slice *roi* using the inset values in *crop*.

        Returns ``(cropped_roi, (x1, y1, x2, y2))`` where the rect is in
        full-ROI-local coordinates.  When all insets are 0 the full ROI is
        returned and the rect is ``(0, 0, w, h)``.
        """
        h, w = roi.shape[:2]
        x1 = max(0, min(crop.left, w - 1))
        y1 = max(0, min(crop.top, h - 1))
        x2 = min(w, max(x1 + 1, w - crop.right)) if crop.right > 0 else w
        y2 = min(h, max(y1 + 1, h - crop.bottom)) if crop.bottom > 0 else h
        return roi[y1:y2, x1:x2], (x1, y1, x2, y2)

    @staticmethod
    def crop_rect_for(roi: np.ndarray, crop: CropRegion) -> tuple[int, int, int, int]:
        """Return only the (x1, y1, x2, y2) rect without slicing the image."""
        _, rect = InspectionPipeline.apply_crop(roi, crop)
        return rect

    @staticmethod
    def _offset_detections(
        detections: list[Detection], x1: int, y1: int
    ) -> list[Detection]:
        """Shift crop-local detection coordinates to full-ROI-local space."""
        if x1 == 0 and y1 == 0:
            return detections
        return [
            Detection(
                bbox=(d.bbox[0] + x1, d.bbox[1] + y1, d.bbox[2], d.bbox[3]),
                centroid=(d.centroid[0] + x1, d.centroid[1] + y1),
                area_px=d.area_px,
                width_px=d.width_px,
                height_px=d.height_px,
                confidence=d.confidence,
                label=d.label,
            )
            for d in detections
        ]

    # ---------- SECTION 2: shape / size rejection ----------

    @staticmethod
    def filter_dough_candidates(
        detections: list[Detection],
        profile: ProductProfile,
        detection_cfg: "DetectionSettings",
    ) -> list[Detection]:
        """Reject candidates that don't match the learned dough profile.

        SECTION 2: a real piece must be ROUND (circularity at/above
        ``shape_filter.min_circularity``) AND DOUGH-SIZED (mean diameter within
        ``(1 ± size_tolerance)`` of the reference dough diameter). Reflections
        on metal are bright but fail one or both, so they no longer count as
        dough or fire false stops.

        Size is relative to the LEARNED dough diameter (``expected_width_px``,
        set by Learn Reference), falling back to the median detected diameter
        when the profile has no learned size — never a fixed pixel count, so it
        adapts to the dough size automatically.

        Circularity is only enforced when the detection actually carries a
        measured value (> 0). Hough returns ideal circles (1.0) whose real
        roundness gate is its accumulator threshold ``param2``; blob/contour
        carry their measured outer-contour circularity here.
        """
        sf = detection_cfg.shape_filter
        if not sf.enabled or not detections:
            return detections

        ref_d = float(profile.expected_width_px)
        if ref_d <= 1.0:
            ref_d = median_piece_diameter(detections)
        kept: list[Detection] = []
        for d in detections:
            if 0.0 < d.circularity < sf.min_circularity:
                continue
            if ref_d > 1.0:
                w = float(d.width_px or 0.0)
                h = float(d.height_px or 0.0)
                diam = (w + h) / 2.0 if (w > 0 and h > 0) else max(w, h)
                if diam > 0.0 and not (
                    ref_d * (1.0 - sf.size_tolerance)
                    <= diam
                    <= ref_d * (1.0 + sf.size_tolerance)
                ):
                    continue
            kept.append(d)
        return kept

    # ---------- shared detection dispatch ----------

    @staticmethod
    def detect_cloth_pieces(
        detector: Detector,
        image: np.ndarray,
        profile: ProductProfile,
        detection_cfg: "DetectionSettings",
        *,
        bg_reference: Optional[np.ndarray] = None,
    ) -> list[Detection]:
        """Dispatch to the currently active ``detection_cfg.method``.

        Bug fix: Learn Reference (``MainWindow._on_learn_reference``) used to
        always call ``detector.detect()`` (blob) regardless of the active
        method, so with e.g. Hough active and showing 38 pieces live, Learn
        Reference would re-run blob from scratch and could find 0. This is
        the single place that decision is made, so Learn Reference and the
        calibration page's live preview can never disagree about which
        pieces the active method finds.

        ``image`` should already be cropped to the cloth ROI (caller's
        responsibility — see ``apply_crop``). ``bg_reference`` should be
        cropped the same way when ``method == "bg_subtract"``; ignored
        otherwise. Returns detections in ``image``'s local coordinate space
        (no offset applied).

        Deliberately NOT used by ``run_cloth_tracking`` below — that method
        also needs the learned-reference shape gates (min_circularity_shape/
        min_solidity_shape) which don't apply here: Learn Reference has
        always run blob with zero shape gates (nothing learned yet to gate
        against), and this preserves that for every method, not just blob.
        """
        method = detection_cfg.method
        if method == "contour_external":
            detections, _binary, _scale = detector.detect_contour_external(  # type: ignore[attr-defined]
                image, profile,
                min_circularity=detection_cfg.contour_external.min_circularity,
            )
        elif method == "hough":
            h_cfg = detection_cfg.hough
            detections, _binary, _scale = detector.detect_hough(  # type: ignore[attr-defined]
                image, profile,
                dp=h_cfg.dp, min_dist_px=h_cfg.min_dist_px,
                param1=h_cfg.param1, param2=h_cfg.param2,
                min_radius_px=h_cfg.min_radius_px, max_radius_px=h_cfg.max_radius_px,
                radius_tolerance=h_cfg.radius_tolerance,
                gate_to_cloth=h_cfg.gate_to_cloth,
                cloth_brightness_threshold=h_cfg.cloth_brightness_threshold,
                downscale_factor=h_cfg.downscale_factor,
            )
        elif method == "bg_subtract":
            detections, _binary, _scale = detector.detect_bg_subtract(  # type: ignore[attr-defined]
                image, bg_reference, profile,
                threshold=detection_cfg.bg_subtract.threshold,
            )
        else:  # "blob" — same call learn_reference has always made for blob.
            detections = detector.detect(
                image, profile,
                fill_holes=detection_cfg.fill_mask_holes,
                fill_holes_kernel=detection_cfg.fill_mask_holes_kernel,
            )
        return detections

    # ---------- tracking ----------

    def run_cloth_tracking(
        self,
        frame: np.ndarray,
        profile: ProductProfile,
        detection_cfg: "DetectionSettings",
        *,
        bg_reference_cloth: Optional[np.ndarray] = None,
    ) -> ClothTrackingResult:
        """Detect pieces in the cloth ROI (after cloth_crop).

        ``detection_cfg`` (``AppConfig.detection``, see core/config.py)
        selects which of the four detection methods runs — "blob" (default,
        unchanged behavior), "contour_external", "hough", or "bg_subtract".
        All four are gated on (cropped to) the SAME cloth ROI, so detection
        never runs outside it regardless of method.

        ``bg_reference_cloth`` is the captured empty-cloth reference frame
        (full cloth ROI, pre-crop) used only when
        ``detection_cfg.method == "bg_subtract"``; ignored otherwise.

        Tripwire occupancy is computed by slicing the transfer-line strip from
        the binary mask already produced by the detector — no redundant image
        preprocessing.  That binary is ALWAYS the plain Otsu/adaptive threshold
        (see ClassicalDetector docstring), regardless of ``detection_cfg.method``,
        so switching methods never changes tripwire firing behavior.  Returns
        ``crossed_transfer_line=False`` always; the tripwire edge logic lives
        in MainWindow.
        """
        _belt, cloth = self.split_rois(frame, profile)
        t0 = time.perf_counter()

        cropped, (cx1, cy1, _cx2, _cy2) = self.apply_crop(cloth, profile.cloth_crop)

        # Detection zone: zero out pixels outside [bridge_left, bridge_right +
        # detection_zone_width_px] before any detection method runs. Hough
        # (and all other methods) see only the relevant approach-side strip —
        # far-away reflections and empty cloth areas are invisible to them.
        # bridge_half is derived from profile.expected_width_px (calibrated
        # size) so it matches the runtime size-adaptive bridge after Learn Ref.
        # Tripwire strip (±tripwire_half_width_px ≈ 15 px from transfer line)
        # is always well inside the zone, so tripwire occupancy is unaffected.
        zone_px = detection_cfg.detection_zone_width_px
        if zone_px > 0:
            _t_local = profile.transfer_line_x - profile.roi_split_x
            _bridge_half = float(profile.expected_width_px) / 2.0
            _c_left = max(0, int(_t_local - _bridge_half) - cx1)
            _c_right = min(cropped.shape[1], int(_t_local + _bridge_half + zone_px) - cx1)
            if _c_right > _c_left:
                _zoned = np.zeros_like(cropped)
                _zoned[:, _c_left:_c_right] = cropped[:, _c_left:_c_right]
                cropped = _zoned

        method = detection_cfg.method

        if method == "contour_external":
            detections, _binary, _det_scale = self._detector.detect_contour_external(
                cropped, profile,
                min_circularity=detection_cfg.contour_external.min_circularity,
            )
        elif method == "hough":
            h_cfg = detection_cfg.hough
            detections, _binary, _det_scale = self._detector.detect_hough(
                cropped, profile,
                dp=h_cfg.dp, min_dist_px=h_cfg.min_dist_px,
                param1=h_cfg.param1, param2=h_cfg.param2,
                min_radius_px=h_cfg.min_radius_px, max_radius_px=h_cfg.max_radius_px,
                radius_tolerance=h_cfg.radius_tolerance,
                gate_to_cloth=h_cfg.gate_to_cloth,
                cloth_brightness_threshold=h_cfg.cloth_brightness_threshold,
                downscale_factor=h_cfg.downscale_factor,
            )
        elif method == "bg_subtract":
            cropped_reference = None
            if (
                bg_reference_cloth is not None
                and bg_reference_cloth.shape[:2] == cloth.shape[:2]
            ):
                cropped_reference, _ = self.apply_crop(bg_reference_cloth, profile.cloth_crop)
            detections, _binary, _det_scale = self._detector.detect_bg_subtract(
                cropped, cropped_reference, profile,
                threshold=detection_cfg.bg_subtract.threshold,
            )
        else:  # "blob" — DEFAULT, exact previous behavior.
            min_circ_shape = (
                profile.ref_circularity_min * profile.circularity_tolerance
                if profile.ref_circularity_min > 0 else 0.0
            )
            min_sol_shape = (
                profile.ref_solidity_min * profile.solidity_tolerance
                if profile.ref_solidity_min > 0 else 0.0
            )
            # EXPERIMENTAL (RELAX_SOLIDITY): cloth-only — lower the solidity
            # floor so hollow-ring blobs (specular highlight punching a fake
            # hole) pass the shape gate. Only ever lowers it; never raises a
            # gate the profile left stricter than the floor. Belt's own
            # min_sol_shape calc in run_belt_inspection below is untouched.
            if RELAX_SOLIDITY and min_sol_shape > 0.0:
                min_sol_shape = min(min_sol_shape, RELAXED_MIN_SOLIDITY_SHAPE)

            # Use detect_with_binary when available (ClassicalDetector) so the
            # tripwire can reuse the threshold binary instead of re-running
            # gray-conversion + blur + Otsu on the full ROI a second time.
            if hasattr(self._detector, "detect_with_binary"):
                detections, _binary, _det_scale = (
                    self._detector.detect_with_binary(  # type: ignore[attr-defined]
                        cropped, profile,
                        min_circularity_shape=min_circ_shape,
                        min_solidity_shape=min_sol_shape,
                        fill_holes=detection_cfg.fill_mask_holes,
                        fill_holes_kernel=detection_cfg.fill_mask_holes_kernel,
                    )
                )
            else:
                detections = self._detector.detect(
                    cropped, profile,
                    min_circularity_shape=min_circ_shape,
                    min_solidity_shape=min_sol_shape,
                    fill_holes=detection_cfg.fill_mask_holes,
                    fill_holes_kernel=detection_cfg.fill_mask_holes_kernel,
                )
                _binary, _det_scale = None, 1.0

        t_det = time.perf_counter()
        # SECTION 2: drop non-dough candidates (reflections/streaks of the
        # wrong shape or size) before they are drawn or drive any stop. The
        # tripwire below reads the raw threshold binary, NOT this list, so the
        # legacy tripwire firing behavior is unaffected by this filter.
        detections = self.filter_dough_candidates(detections, profile, detection_cfg)
        detections = self._offset_detections(detections, cx1, cy1)
        tripwire_occ, tripwire_hit = self._compute_tripwire(
            cropped, profile, cx1, _binary, _det_scale
        )
        t_trip = time.perf_counter()

        det_ms = (t_det - t0) * 1000.0
        trip_ms = (t_trip - t_det) * 1000.0
        elapsed = (t_trip - t0) * 1000.0
        logger.debug(
            "cloth_tracking: detection={:.1f}ms tripwire={:.1f}ms total={:.1f}ms",
            det_ms, trip_ms, elapsed,
        )

        # Throttled (not per-frame) summary of the active method + piece count,
        # so A/B comparisons between methods are visible in the log.
        _now_method_log = time.monotonic()
        if _now_method_log - self._last_method_log_time >= 2.0:
            self._last_method_log_time = _now_method_log
            logger.info(
                "cloth detection: method={} pieces_in_roi={}", method, len(detections),
            )

        # --- DIAGNOSTIC: row-profile projection (two-rows-at-once investigation) ---
        # Reuses the binary mask already computed above for detection/tripwire —
        # no extra preprocessing. A single np.sum over an existing array, so the
        # added per-frame cost is negligible. Purely descriptive: nothing below
        # feeds back into detections, tripwire_occ/hit, or any verdict.
        row_profile = None
        row_profile_scale = 1.0
        if DIAGNOSTIC_ROW_PROFILE and _binary is not None:
            row_profile = _binary.sum(axis=0).astype(np.float64)
            row_profile_scale = _det_scale
        # --- END DIAGNOSTIC ---

        front = [c.centroid for c in detections]
        return ClothTrackingResult(
            front_row_centroids=front,
            crossed_transfer_line=False,
            detections=detections,
            inference_ms=elapsed,
            crop_rect=self.crop_rect_for(cloth, profile.cloth_crop),
            tripwire_occupancy=tripwire_occ,
            tripwire_occupied=tripwire_hit,
            row_profile=row_profile,
            row_profile_scale=row_profile_scale,
        )

    # ---------- tripwire ----------

    # Shadow floor / adaptive constants match ClassicalDetector defaults.
    # Only used in the strip-first fallback path.
    _TRIPWIRE_SHADOW_FLOOR: int = 40

    @staticmethod
    def _compute_tripwire(
        cropped: np.ndarray,
        profile: "ProductProfile",
        crop_x_offset: int,
        binary: "np.ndarray | None" = None,
        binary_scale: float = 1.0,
    ) -> tuple[float, bool]:
        """Return (occupancy_fraction, fired) for the transfer-line strip.

        Fast path (binary provided): slices the strip from the threshold binary
        already produced by the detector — zero redundant preprocessing.

        Fallback (binary=None): processes only the thin strip column, not the
        full ROI, so overhead is ~30×H pixels instead of W×H.
        """
        line_x = profile.transfer_line_x - profile.roi_split_x - crop_x_offset
        half_w = profile.tripwire_half_width_px

        if binary is not None:
            # --- Fast path: reuse detector's threshold binary ---
            # Map strip coords to the (possibly downscaled) binary space.
            bx0 = max(0, int((line_x - half_w) * binary_scale))
            bx1 = min(binary.shape[1], int((line_x + half_w) * binary_scale) + 1)
            if bx0 >= bx1:
                return 0.0, False
            strip = binary[:, bx0:bx1]
            occupancy = float(np.count_nonzero(strip)) / strip.size
            return occupancy, occupancy >= profile.tripwire_occupancy_threshold

        # --- Fallback: strip-first, full threshold logic on the thin column only ---
        if cropped.size == 0:
            return 0.0, False

        x0 = max(0, line_x - half_w)
        x1 = min(cropped.shape[1], line_x + half_w)
        if x0 >= x1:
            return 0.0, False

        strip_raw = cropped[:, x0:x1]
        if strip_raw.ndim == 3 and strip_raw.shape[2] == 4:
            gray = cv2.cvtColor(strip_raw, cv2.COLOR_BGRA2GRAY)
        elif strip_raw.ndim == 3:
            gray = cv2.cvtColor(strip_raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = strip_raw
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        sf = InspectionPipeline._TRIPWIRE_SHADOW_FLOOR
        if profile.dough_is_darker:
            shadow = blurred < sf
            masked = blurred.copy()
            masked[shadow] = 255
            non_shadow = ~shadow
            std = float(np.std(masked[non_shadow])) if non_shadow.any() else 0.0
            if std < 3.0:
                bsz = max(3, min(51, gray.shape[0])) | 1
                dough_bin = cv2.adaptiveThreshold(
                    masked, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, bsz, 10,
                )
                dough_bin[shadow] = 0
            else:
                _, dough_bin = cv2.threshold(
                    masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                dough_bin[shadow] = 0
        else:
            std = float(np.std(blurred))
            if std < 3.0:
                bsz = max(3, min(51, gray.shape[0])) | 1
                dough_bin = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, bsz, -10,
                )
            else:
                _, dough_bin = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

        occupancy = float(np.count_nonzero(dough_bin)) / dough_bin.size
        return occupancy, occupancy >= profile.tripwire_occupancy_threshold

    # ---------- DIAGNOSTIC: row-profile analysis (two-rows-at-once investigation) ----------
    # Purely descriptive — return value is only ever logged/drawn, never fed
    # back into a decision. Safe to delete along with DIAGNOSTIC_ROW_PROFILE.

    @staticmethod
    def analyze_row_profile(
        profile: np.ndarray,
        *,
        smoothing: int = 5,
        bump_threshold_frac: float = 0.25,
    ) -> tuple[int, float]:
        """Return (bump_count, deepest_valley_pct) for a column-sum profile.

        A "bump" is a contiguous run of columns at/above ``bump_threshold_frac``
        of the profile's peak — one queued row should show as one bump, two
        rows separated by a visible gap should show as two. ``deepest_valley_pct``
        is the largest drop (as % of peak) found between two adjacent bumps.
        """
        if profile.size == 0:
            return 0, 0.0

        if smoothing > 1:
            kernel = np.ones(smoothing, dtype=np.float64) / smoothing
            smoothed = np.convolve(profile.astype(np.float64), kernel, mode="same")
        else:
            smoothed = profile.astype(np.float64)

        peak = float(smoothed.max())
        if peak <= 0:
            return 0, 0.0
        threshold = bump_threshold_frac * peak

        above = smoothed >= threshold
        runs: list[tuple[int, int]] = []
        start = 0
        in_run = False
        for i, v in enumerate(above):
            if v and not in_run:
                in_run, start = True, i
            elif not v and in_run:
                in_run = False
                runs.append((start, i))
        if in_run:
            runs.append((start, len(above)))

        deepest_valley_pct = 0.0
        for (_, end_a), (start_b, _) in zip(runs, runs[1:]):
            if start_b > end_a:
                valley_min = float(smoothed[end_a:start_b].min())
                deepest_valley_pct = max(
                    deepest_valley_pct, max(0.0, (peak - valley_min) / peak * 100.0)
                )

        return len(runs), deepest_valley_pct

    # ---------- inspecting ----------

    def run_belt_inspection(
        self,
        frame: np.ndarray,
        profile: ProductProfile,
        *,
        unknown_is_fault: bool = False,
    ) -> PipelineResult:
        """Inspect the belt ROI (after belt_crop). Returned detection coords are
        in belt-ROI space (not crop-local space).

        Belt detection uses ``profile.belt_dough_is_darker`` (default False)
        rather than ``profile.dough_is_darker``.  The belt is typically a dark
        wire-mesh / grating with lighter dough on top, which is the OPPOSITE
        contrast to the cloth (bright cloth, dark dough).

        ``unknown_is_fault`` (config ``inspection.unknown_is_fault``, FIX 6):
        when False (default), an "unknown"-labeled blob is NOT an automatic
        hard fault — see :meth:`_verdict_from_row`.
        """
        belt, _cloth = self.split_rois(frame, profile)
        t0 = time.perf_counter()

        cropped, (cx1, cy1, _cx2, _cy2) = self.apply_crop(belt, profile.belt_crop)
        # Use belt-specific polarity; do not inherit the cloth polarity setting.
        belt_detect_profile = profile.model_copy(
            update={"dough_is_darker": profile.belt_dough_is_darker}
        )
        # Adaptive threshold block size for belt detection.
        # profile.belt_adaptive_block = 0 → auto: 2 × expected width (odd).
        # Adaptive threshold compares each pixel to its local neighbourhood mean
        # rather than a global Otsu value.  This prevents adjacent donuts on an
        # unevenly-lit belt from merging into a single large blob.
        adaptive_block = profile.belt_adaptive_block
        if adaptive_block == 0 and profile.expected_width_px > 20:
            adaptive_block = max(51, int(profile.expected_width_px * 2)) | 1
        min_circ_shape = (
            profile.ref_circularity_min * profile.circularity_tolerance
            if profile.ref_circularity_min > 0 else 0.0
        )
        min_sol_shape = (
            profile.ref_solidity_min * profile.solidity_tolerance
            if profile.ref_solidity_min > 0 else 0.0
        )
        detections = self._detector.detect(
            cropped, belt_detect_profile,
            force_adaptive_block=adaptive_block,
            min_solidity=profile.belt_min_solidity,
            max_aspect_ratio=profile.belt_max_aspect_ratio,
            min_circularity_shape=min_circ_shape,
            min_solidity_shape=min_sol_shape,
        )
        detections = self._offset_detections(detections, cx1, cy1)
        detections = self._proximity_merge(detections, profile)

        elapsed = (time.perf_counter() - t0) * 1000.0

        newest_row = self._newest_row(detections, profile)
        verdict, reason = self._verdict_from_row(
            newest_row, profile, unknown_is_fault=unknown_is_fault
        )

        logger.debug(
            "Belt inspection: {} blob(s) in ROI, {} in newest row, "
            "verdict={}, crop=({},{},{},{}), belt_dough_is_darker={}, {:.1f} ms",
            len(detections), len(newest_row), verdict.value,
            cx1, cy1, _cx2, _cy2, profile.belt_dough_is_darker, elapsed,
        )
        for i, d in enumerate(detections[:8]):
            logger.debug(
                "  blob {}: bbox={} area={} w={} h={} label={}",
                i + 1, d.bbox, d.area_px, d.width_px, d.height_px, d.label,
            )

        return PipelineResult(
            verdict=verdict,
            detections=detections,
            inference_ms=elapsed,
            fault_reason=reason,
            crop_rect=self.crop_rect_for(belt, profile.belt_crop),
        )

    # ---------- internals ----------

    @staticmethod
    def _newest_row(
        detections: list[Detection], profile: ProductProfile
    ) -> list[Detection]:
        """Pick the rightmost cluster of centroids (the row just deposited)."""
        if not detections:
            return []
        xs = [d.centroid[0] for d in detections]
        max_x = max(xs)
        band = 0.7 * profile.expected_width_px
        return [d for d in detections if d.centroid[0] >= max_x - band]

    @staticmethod
    def _proximity_merge(
        detections: list[Detection], profile: ProductProfile
    ) -> list[Detection]:
        """Merge blobs whose centroids are within fused_merge_distance_factor × diameter.

        Detects stuck pieces that the threshold split into two blobs.  The
        merged detection is re-classified by geometry so touching Berliners
        become row_fused / column_fused as expected.
        """
        if len(detections) < 2:
            return detections
        diameter = (profile.expected_width_px + profile.expected_height_px) / 2.0
        threshold = profile.fused_merge_distance_factor * diameter

        result = list(detections)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    di, dj = result[i], result[j]
                    dx = di.centroid[0] - dj.centroid[0]
                    dy = di.centroid[1] - dj.centroid[1]
                    if (dx * dx + dy * dy) ** 0.5 < threshold:
                        x1 = min(di.bbox[0], dj.bbox[0])
                        y1 = min(di.bbox[1], dj.bbox[1])
                        x2 = max(di.bbox[0] + di.bbox[2], dj.bbox[0] + dj.bbox[2])
                        y2 = max(di.bbox[1] + di.bbox[3], dj.bbox[1] + dj.bbox[3])
                        mw, mh = x2 - x1, y2 - y1
                        tot = di.area_px + dj.area_px
                        mcx = (di.centroid[0] * di.area_px + dj.centroid[0] * dj.area_px) / tot
                        mcy = (di.centroid[1] * di.area_px + dj.centroid[1] * dj.area_px) / tot
                        merged = Detection(
                            bbox=(x1, y1, mw, mh),
                            centroid=(mcx, mcy),
                            area_px=tot,
                            width_px=mw,
                            height_px=mh,
                            confidence=min(1.0, tot / max(profile.expected_area_px, 1)),
                            label=_ClassicalDetector._classify(mw, mh, profile),
                        )
                        result = [result[k] for k in range(len(result)) if k not in (i, j)]
                        result.append(merged)
                        changed = True
                        break
                if changed:
                    break
        return result

    @staticmethod
    def _unknown_orientation_verdict(
        d: Detection, profile: ProductProfile
    ) -> tuple[Verdict, str]:
        """FIX 6: best-effort row/column orientation read for a blob
        ``ClassicalDetector._classify`` couldn't confidently label
        single/fused (size out of bounds, or wide+tall together).

        Only a clearly wide-dominant (row-wise / horizontal contact) blob
        escalates to a hard fault — that's the same physical defect
        ``row_fused`` describes, just one ``_classify`` didn't cleanly bucket.
        A tall-dominant (column-wise) or genuinely ambiguous blob is
        informational only, exactly like ``column_fused`` — it never raises
        FaultActive or writes ext_error. Does NOT change ``_classify`` or any
        detection threshold; this only affects which Verdict an already-
        produced "unknown" label maps to.
        """
        w_ratio = d.width_px / max(profile.expected_width_px, 1)
        h_ratio = d.height_px / max(profile.expected_height_px, 1)
        if w_ratio >= profile.fused_threshold and w_ratio >= h_ratio:
            return Verdict.FAULT_ROW_FUSED, f"unknown blob read as row-wise at {d.centroid}"
        return (
            Verdict.INFO_UNKNOWN_NONFAULT,
            f"unknown blob at {d.centroid} (not row-wise; informational only)",
        )

    @staticmethod
    def _verdict_from_row(
        row: list[Detection], profile: ProductProfile, *, unknown_is_fault: bool = False,
    ) -> tuple[Verdict, str]:
        """``unknown_is_fault`` (FIX 6, config ``inspection.unknown_is_fault``):
        False (default) routes "unknown" blobs through
        :meth:`_unknown_orientation_verdict` instead of an immediate hard
        fault. True restores the old behavior (every "unknown" blob is an
        immediate ``FAULT_UNKNOWN``).
        """
        if not row:
            return Verdict.INFO_INCOMPLETE_ROW, "no detections in newest row"
        for d in row:
            if d.label == "unknown":
                if unknown_is_fault:
                    return Verdict.FAULT_UNKNOWN, f"unknown blob at {d.centroid}"
                verdict, reason = InspectionPipeline._unknown_orientation_verdict(d, profile)
                if verdict.is_fault:
                    return verdict, reason
                # Column-wise or ambiguous — informational only; keep scanning
                # the row, since another blob in it might still be a real fault.
                continue
            if d.label == "row_fused":
                return Verdict.FAULT_ROW_FUSED, f"row_fused at {d.centroid}"
        for d in row:
            if d.label == "column_fused":
                return Verdict.INFO_COLUMN_FUSED, ""
            if d.label == "unknown":  # unknown_is_fault is False here (handled above otherwise)
                return Verdict.INFO_UNKNOWN_NONFAULT, f"unknown blob at {d.centroid}"
        return Verdict.OK, ""
