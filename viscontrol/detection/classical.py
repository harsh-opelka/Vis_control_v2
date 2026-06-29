"""Classical OpenCV detector.

Pipeline (all on the input ROI in its already-rotated orientation):
    1. Convert to single-channel grayscale if needed.
    2. Optionally downscale to ``max_detect_width`` for speed (detections are
       scaled back to original coordinates before returning).
    3. Light Gaussian blur to suppress sensor noise.
    4. Threshold.  When ``profile.dough_is_darker`` is True (dark dough on bright
       cloth):
         a. Shadow-mask: replace every pixel below ``shadow_floor`` with 255 so
            the deep-black machine frame is treated as bright background by Otsu.
            Without this step Otsu splits {machine frame} vs {cloth + donuts}
            instead of {cloth} vs {donuts}, and the frame becomes foreground.
         b. Otsu with THRESH_BINARY_INV on the masked image → dark donuts become
            foreground; bright cloth and shadow areas become background.
         c. Force shadow pixels to 0 in the result so they can never be reported
            as dough.
       When ``profile.dough_is_darker`` is False (bright dough on dark belt):
         Otsu with THRESH_BINARY — original behaviour.
       Both modes fall back to adaptive threshold when image variance is too low
       for Otsu to produce a useful split.
    5. Morphological open with a small ellipse to drop salt-and-pepper specks.
    6. ``findContours(RETR_EXTERNAL)`` to enumerate blobs — outer contours only,
       so ring/donut shapes (with a central hole) are detected as a single piece
       with the correct outer-diameter bounding box.
    7. Drop blobs below ``profile.noise_threshold * profile.expected_area_px``
       (minimum size) or above ``max_roi_fraction`` of the total ROI area
       (maximum size — rejects machine-frame blobs that are far larger than any
       dough piece).
    8. Classify each remaining blob using bounding-box width/height vs profile:
       - width  > expected_width  * fused_threshold -> "column_fused"
       - height > expected_height * fused_threshold -> "row_fused"
       - both wildly different                      -> "unknown"
       - else                                       -> "single"

EXPERIMENTAL (ring/hollow-dough detection, controlled by pipeline.py's
RELAX_SOLIDITY and config detection.fill_mask_holes, cloth-only): step 6
normally already measures area/solidity/circularity from the OUTER contour
only, so a clean ring with a fully interior hole already has near-identical
metrics to a solid disk (the hole itself is invisible to RETR_EXTERNAL).
Rejection in practice usually means the highlight also notches/breaks the
outer rim, genuinely lowering solidity/circularity. ``_fill_holes`` (closes
small breaks, then fills any still-enclosed hole) and a lowered
``min_solidity_shape`` floor in ``_extract`` are both there to recover from
that. See :meth:`ClassicalDetector._fill_holes`.

ALTERNATIVE DETECTION METHODS (config ``detection.method``, cloth-only — see
viscontrol/detection/pipeline.py's ``run_cloth_tracking``): the methods above
are "blob" (this docstring), the default. Three more live alongside it,
selectable without touching this method's behavior at all:

  - ``detect_contour_external``: same threshold pipeline as blob, but the
    extraction step (``_extract_external_only``) gates on area + circularity
    of the OUTER contour only — no solidity check at all. A hollow ring with
    an intact outer rim keeps near-circular outer-contour metrics and passes;
    only the (already-disabled) solidity gate was rejecting it in blob mode.
  - ``detect_hough``: ``cv2.HoughCircles`` on the grayscale image directly —
    finds circular edges regardless of what's inside them, so a shiny dome's
    fake interior ring is irrelevant to begin with.
  - ``detect_bg_subtract``: pixel-diff against a captured empty-cloth
    reference frame — "whatever changed is dough", independent of absolute
    brightness/threshold entirely.

All four return the same ``(list[Detection], binary, scale)`` shape used by
``detect_with_binary``, and the returned ``binary`` is always the same raw
pre-morph Otsu/adaptive threshold used by the tripwire — switching
``detection.method`` never changes the tripwire's firing behavior, only what
the *piece list* contains.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from viscontrol.core.profiles import ProductProfile
from viscontrol.detection.base import Detection


class ClassicalDetector:
    """OpenCV-based dough blob detector."""

    def __init__(
        self,
        *,
        blur_ksize: int = 5,
        morph_ksize: int = 5,
        adaptive_block: int = 51,
        adaptive_C: int = -10,
        # --- scene constraints ---
        shadow_floor: int = 40,
        # pixels strictly below this value are treated as deep shadow /
        # machine frame and forced to background before Otsu runs.
        # Prevents the 3-level scene (black frame / gray donuts / white cloth)
        # from confusing Otsu into splitting at the frame boundary.
        max_roi_fraction: float = 0.15,
        # blobs larger than this fraction of the total image area are rejected
        # (machine frame, gaps, stray large reflections).
        # --- performance ---
        max_detect_width: int = 800,
        # downscale input to this width before processing; detections are scaled
        # back to original coordinates.  Set to 0 to disable downscaling.
    ) -> None:
        if blur_ksize % 2 == 0 or blur_ksize < 1:
            raise ValueError("blur_ksize must be odd and >= 1")
        if morph_ksize < 1:
            raise ValueError("morph_ksize must be >= 1")
        self._blur_ksize = blur_ksize
        self._morph_ksize = morph_ksize
        self._adaptive_block = adaptive_block
        self._adaptive_C = adaptive_C
        self._shadow_floor = shadow_floor
        self._max_roi_fraction = max_roi_fraction
        self._max_detect_width = max_detect_width

    # ---------- public ----------

    def detect(
        self,
        image: np.ndarray,
        profile: ProductProfile,
        *,
        force_adaptive_block: int = 0,
        min_solidity: float = 0.0,
        max_aspect_ratio: float = 0.0,
        min_circularity_shape: float = 0.0,
        min_solidity_shape: float = 0.0,
        fill_holes: bool = False,
        fill_holes_kernel: int = 9,
    ) -> list[Detection]:
        """Detect blobs in *image* using *profile* parameters.

        ``force_adaptive_block`` overrides the Otsu threshold with an adaptive
        threshold using the given block size.  Used by ``run_belt_inspection``
        which passes a block size derived from the expected donut diameter to
        prevent nearby donuts from merging into one blob due to uneven belt
        lighting.  Must be odd and > 1; 0 = no override (use Otsu or shadow-
        masked Otsu as normal).

        ``min_solidity`` and ``max_aspect_ratio`` are belt-specific pre-filter
        gates (applied before shape checks).  0.0 means disabled.

        ``min_circularity_shape`` and ``min_solidity_shape`` are learned-
        reference shape gates (applied to both belt and cloth).  0.0 = disabled.
        Circularity gate is skipped for blobs large enough to be fused pairs.

        ``fill_holes``/``fill_holes_kernel`` — EXPERIMENTAL (ring/hollow-dough
        detection), see :meth:`_fill_holes`. Default False/unused: callers that
        don't pass it get exactly the previous behavior.
        """
        return self._detect_full(
            image, profile,
            force_adaptive_block=force_adaptive_block,
            min_solidity=min_solidity,
            max_aspect_ratio=max_aspect_ratio,
            min_circularity_shape=min_circularity_shape,
            min_solidity_shape=min_solidity_shape,
            fill_holes=fill_holes,
            fill_holes_kernel=fill_holes_kernel,
        )[0]

    def detect_with_binary(
        self,
        image: np.ndarray,
        profile: ProductProfile,
        *,
        force_adaptive_block: int = 0,
        min_solidity: float = 0.0,
        max_aspect_ratio: float = 0.0,
        min_circularity_shape: float = 0.0,
        min_solidity_shape: float = 0.0,
        fill_holes: bool = False,
        fill_holes_kernel: int = 9,
    ) -> tuple[list[Detection], np.ndarray, float]:
        """Like ``detect()`` but also returns ``(thresh_binary, scale_factor)``.

        ``thresh_binary`` is the raw threshold result (dough=255, background=0)
        at the detector's internal (possibly downscaled) resolution, BEFORE
        morphological open/dilation AND before any ``fill_holes`` processing —
        this is exactly what the tripwire slices for occupancy counting, and
        that must stay unaffected by the experimental hole-fill. ``scale_factor``
        maps original-image x/y coordinates to binary coordinates:
        ``binary_x = orig_x * scale``.

        Used by the tripwire to slice the transfer-line strip from the already-
        computed binary instead of re-running preprocessing on the full image.
        """
        return self._detect_full(
            image, profile,
            force_adaptive_block=force_adaptive_block,
            min_solidity=min_solidity,
            max_aspect_ratio=max_aspect_ratio,
            min_circularity_shape=min_circularity_shape,
            min_solidity_shape=min_solidity_shape,
            fill_holes=fill_holes,
            fill_holes_kernel=fill_holes_kernel,
        )

    def _detect_full(
        self,
        image: np.ndarray,
        profile: ProductProfile,
        *,
        force_adaptive_block: int = 0,
        min_solidity: float = 0.0,
        max_aspect_ratio: float = 0.0,
        min_circularity_shape: float = 0.0,
        min_solidity_shape: float = 0.0,
        fill_holes: bool = False,
        fill_holes_kernel: int = 9,
    ) -> tuple[list[Detection], np.ndarray, float]:
        """Core implementation shared by ``detect`` and ``detect_with_binary``."""
        if image.size == 0:
            return [], np.zeros((1, 1), dtype=np.uint8), 1.0

        h, w = image.shape[:2]
        scale = 1.0
        if self._max_detect_width and w > self._max_detect_width:
            scale = self._max_detect_width / w
            new_h = max(1, int(h * scale))
            image = cv2.resize(
                image, (self._max_detect_width, new_h), interpolation=cv2.INTER_AREA
            )

        # Scale profile dimensions to match the (possibly downscaled) image so
        # the min-area filter operates in the correct pixel space.
        extract_profile = profile
        if scale < 1.0:
            scale2 = scale * scale
            extract_profile = profile.model_copy(
                update={
                    "expected_area_px": max(1, int(profile.expected_area_px * scale2)),
                    "expected_width_px": max(1, int(profile.expected_width_px * scale)),
                    "expected_height_px": max(1, int(profile.expected_height_px * scale)),
                }
            )

        # Scale the forced adaptive block too (it's in original-image coords).
        scaled_force_block = 0
        if force_adaptive_block > 1:
            scaled_force_block = max(3, int(force_adaptive_block * scale)) | 1

        gray = self._to_gray(image)
        blurred = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)
        binary = self._threshold(blurred, invert=profile.dough_is_darker,
                                 force_adaptive_block=scaled_force_block)
        # EXPERIMENTAL (FILL_MASK_HOLES, see pipeline.py): operates on a
        # *separate* mask used only for blob extraction below. `binary` itself
        # — returned as-is at the end of this method — stays the raw,
        # unfilled threshold result, since that's what the tripwire slices for
        # occupancy counting and must not change.
        extract_source = (
            self._fill_holes(binary, fill_holes_kernel) if fill_holes else binary
        )
        opened = self._morph_open(extract_source)
        dilation_kernel_size = profile.dilation_kernel_size
        if scale < 1.0:
            dilation_kernel_size = max(0, int(dilation_kernel_size * scale))
        if dilation_kernel_size > 0:
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
            )
            opened = cv2.dilate(opened, dilation_kernel, iterations=1)
        detections = self._extract(
            opened, extract_profile,
            min_solidity=min_solidity,
            max_aspect_ratio=max_aspect_ratio,
            min_circularity_shape=min_circularity_shape,
            min_solidity_shape=min_solidity_shape,
        )

        # Scale detections back to original (full-resolution) coordinates.
        if scale < 1.0:
            inv = 1.0 / scale
            detections = [
                Detection(
                    bbox=(
                        int(d.bbox[0] * inv),
                        int(d.bbox[1] * inv),
                        int(d.bbox[2] * inv),
                        int(d.bbox[3] * inv),
                    ),
                    centroid=(d.centroid[0] * inv, d.centroid[1] * inv),
                    area_px=int(d.area_px * inv * inv),
                    width_px=int(d.width_px * inv),
                    height_px=int(d.height_px * inv),
                    confidence=d.confidence,
                    label=d.label,
                    circularity=d.circularity,
                    solidity=d.solidity,
                )
                for d in detections
            ]
        # Return binary before morph-open/dilation: raw pixel-level threshold
        # result, needed by the tripwire to count individual dough pixels.
        return detections, binary, scale

    def compute_binary_mask(
        self,
        image: np.ndarray,
        profile: ProductProfile,
        *,
        fill_holes: bool = False,
        fill_holes_kernel: int = 9,
    ) -> tuple[np.ndarray, int]:
        """Return (post-morph binary mask, Otsu threshold value) for visualization.

        Read-only — does not affect detection results.  The threshold is the Otsu
        level used on the (possibly shadow-masked) blurred image, which is the same
        split point shown as a vertical line on a brightness histogram.

        ``fill_holes``/``fill_holes_kernel`` — EXPERIMENTAL (FILL_MASK_HOLES,
        see pipeline.py): pass through so the wizard's "Cloth Binary Mask"
        preview can show the filled result. Default False/unused everywhere
        else (e.g. the belt debug-mask thumbnail in main_window).
        """
        if image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), 128

        h, w = image.shape[:2]
        if self._max_detect_width and w > self._max_detect_width:
            scale = self._max_detect_width / w
            new_h = max(1, int(h * scale))
            image = cv2.resize(
                image, (self._max_detect_width, new_h), interpolation=cv2.INTER_AREA
            )

        gray = self._to_gray(image)
        blurred = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)

        if profile.dough_is_darker:
            shadow_mask = blurred < self._shadow_floor
            masked = blurred.copy()
            masked[shadow_mask] = 255
            thresh_f, _ = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thresh_f, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary = self._threshold(blurred, invert=profile.dough_is_darker)
        mask_source = self._fill_holes(binary, fill_holes_kernel) if fill_holes else binary
        return self._morph_open(mask_source), int(thresh_f)

    # ---------- alternative detection methods (config detection.method) ----------

    def detect_contour_external(
        self,
        image: np.ndarray,
        profile: ProductProfile,
        *,
        min_circularity: float = 0.55,
    ) -> tuple[list[Detection], np.ndarray, float]:
        """ALTERNATIVE METHOD (detection.method="contour_external").

        Same gray/blur/threshold pipeline as the default blob method, but
        extraction (``_extract_external_only``) gates ONLY on area and outer-
        contour circularity — no solidity check. A hollow ring (real outer
        rim, fake interior hole from a specular highlight) keeps a near-
        circular outer contour and passes; only blob's solidity gate, not the
        thresholding itself, was ever rejecting it.

        Returns ``(detections, binary, scale)`` — ``binary`` is the same raw
        pre-morph threshold the tripwire uses, so switching to this method
        cannot change tripwire firing behavior.
        """
        if image.size == 0:
            return [], np.zeros((1, 1), dtype=np.uint8), 1.0
        resized, scale = self._resize_for_detect(image)
        extract_profile = self._scaled_profile(profile, scale)
        gray = self._to_gray(resized)
        blurred = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)
        binary = self._threshold(blurred, invert=profile.dough_is_darker)
        opened = self._morph_open(binary)
        detections = self._extract_external_only(
            opened, extract_profile, min_circularity=min_circularity
        )
        detections = self._scale_detections(detections, scale)
        return detections, binary, scale

    def detect_hough(
        self,
        image: np.ndarray,
        profile: ProductProfile,
        *,
        dp: float = 1.2,
        min_dist_px: int = 0,
        param1: float = 80.0,
        param2: float = 45.0,
        min_radius_px: int = 0,
        max_radius_px: int = 0,
        radius_tolerance: float = 0.2,
        gate_to_cloth: bool = True,
        cloth_brightness_threshold: int = 120,
        downscale_factor: float = 1.0,
    ) -> tuple[list[Detection], np.ndarray, float]:
        """ALTERNATIVE METHOD (detection.method="hough").

        ``cv2.HoughCircles`` finds circular edges directly on the grayscale
        image — immune to the hollow-center problem since it never depends on
        a binary fill/solidity test at all. ``min_dist_px``/``min_radius_px``/
        ``max_radius_px`` of 0 auto-derive from ``profile``'s expected piece
        diameter, tightened to 0.8x-1.2x of the expected radius (scaled along
        with the rest of the pipeline).

        FIX 1/3 (``gate_to_cloth``, default True — HOUGH_GATE_TO_CLOTH):
        dough only ever sits on the bright cloth (Gärtuch), never on the
        darker metal grating/mesh/frame. ``cloth_brightness_threshold`` splits
        the two — but NOT via a raw per-pixel threshold, since the dough
        itself is dark (``dough_is_darker``) and a naive pixel test would
        blank out the dough right along with the metal. Instead brightness is
        measured on a morphologically-closed version of the image (see
        ``_cloth_basis``, kernel ~4x the expected piece radius): a single
        dough piece is genuinely erased by closing and recovers its true
        bright cloth level, while a genuinely large metal/grating region
        (much bigger than the kernel) stays dark. The non-cloth region is
        then replaced with that same smooth basis value before Hough runs
        (grating/mesh texture is extremely edge-dense, so this is also the
        dominant fix for the ~637 ms/frame cost — Hough's internal
        Canny+accumulator cost scales with edge-pixel count; filling with the
        basis rather than blanking to 0 also avoids inventing a sharp
        artificial edge along the seam) and any surviving circle whose center
        falls outside the cloth mask is rejected as a second-layer check.

        FIX 3 (``downscale_factor``, default 1.0 = disabled): an additional
        downscale applied ONLY to the Hough step, on top of the shared
        ``max_detect_width`` downscale every method gets. Circle coordinates
        are scaled back up before being returned.

        Returns ``(detections, binary, scale)``; ``binary`` is the ordinary
        Otsu/adaptive threshold (tripwire-compatible) — Hough itself never
        touches it, and neither ``gate_to_cloth`` nor ``downscale_factor``
        affect it.
        """
        if image.size == 0:
            return [], np.zeros((1, 1), dtype=np.uint8), 1.0
        resized, scale = self._resize_for_detect(image)
        extract_profile = self._scaled_profile(profile, scale)
        gray = self._to_gray(resized)
        blurred = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)
        # Kept ONLY for the tripwire-compatible return value; Hough below
        # never touches this — not `blurred`, not `hough_input`.
        binary = self._threshold(blurred, invert=profile.dough_is_darker)

        # FIX 3: extra downscale for the Hough step only. `extra_scale` maps
        # `resized`-space coordinates to `hough_input`-space; everything
        # Hough-related below works in `hough_input`-space, then converts
        # back to `resized`-space before the shared _scale_detections() call.
        hough_input = blurred
        extra_scale = 1.0
        if downscale_factor > 1.0:
            extra_scale = 1.0 / downscale_factor
            new_w = max(1, int(blurred.shape[1] * extra_scale))
            new_h = max(1, int(blurred.shape[0] * extra_scale))
            hough_input = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)

        expected_radius = (
            extract_profile.expected_width_px + extract_profile.expected_height_px
        ) / 4.0 * extra_scale
        combined_scale = scale * extra_scale

        # FIX 1/3: replace the non-cloth (metal/grating/reflective) region
        # with its own smooth local-average value so Canny finds no texture
        # edges there at all — dough can never be there physically, and this
        # is what actually kills the frame cost. Filling with the smooth
        # basis (rather than blanking to 0) avoids introducing a brand-new
        # sharp artificial edge along the cloth/metal seam, which would
        # otherwise interfere with real dough circles sitting close to it.
        cloth_mask: np.ndarray | None = None
        if gate_to_cloth:
            basis = self._cloth_basis(hough_input, expected_radius)
            cloth_mask = self._cloth_mask_from_basis(basis, cloth_brightness_threshold)
            hough_input = np.where(cloth_mask > 0, hough_input, basis).astype(np.uint8)
        min_dist = (
            float(min_dist_px * combined_scale) if min_dist_px > 0
            else max(1.0, expected_radius * 1.6)
        )
        # SECTION 2 (was FIX 2): a real dough piece is a known size, so the
        # auto radius band is (1 - tol)…(1 + tol) × the expected radius. A
        # tighter band both rejects more false circles (reflections/streaks of
        # the wrong size) and cuts the radius search space (speed). ``tol`` is
        # config.detection.hough.radius_tolerance (default 0.2 → 0.8…1.2,
        # preserving the previous hardcoded band).
        tol = max(0.01, min(0.99, radius_tolerance))
        min_r = (
            int(min_radius_px * combined_scale) if min_radius_px > 0
            else max(1, int(expected_radius * (1.0 - tol)))
        )
        max_r = (
            int(max_radius_px * combined_scale) if max_radius_px > 0
            else max(min_r + 1, int(expected_radius * (1.0 + tol)))
        )

        circles = cv2.HoughCircles(
            hough_input, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r,
        )
        detections: list[Detection] = []
        if circles is not None:
            max_area = resized.shape[0] * resized.shape[1] * self._max_roi_fraction
            inv_extra = 1.0 / extra_scale
            for cx, cy, r in circles[0]:
                # FIX 1: reject any circle whose center isn't on the cloth
                # mask — a defensive second layer behind the input blanking
                # above (an edge straddling the cloth/metal boundary could
                # still produce a center that drifts just off it).
                if cloth_mask is not None:
                    my, mx = int(round(cy)), int(round(cx))
                    if (
                        my < 0 or my >= cloth_mask.shape[0]
                        or mx < 0 or mx >= cloth_mask.shape[1]
                        or cloth_mask[my, mx] == 0
                    ):
                        continue
                # Back to `resized`-space (undo the FIX 3 extra downscale).
                cx_r, cy_r, r_r = cx * inv_extra, cy * inv_extra, r * inv_extra
                area = math.pi * r_r * r_r
                if area > max_area:
                    continue
                diameter = int(round(r_r * 2))
                x = int(round(cx_r - r_r))
                y = int(round(cy_r - r_r))
                label = self._classify(diameter, diameter, extract_profile)
                confidence = float(min(1.0, area / max(extract_profile.expected_area_px, 1)))
                detections.append(
                    Detection(
                        bbox=(x, y, diameter, diameter),
                        centroid=(float(cx_r), float(cy_r)),
                        area_px=int(area),
                        width_px=diameter,
                        height_px=diameter,
                        confidence=confidence,
                        label=label,
                        circularity=1.0,
                        solidity=1.0,
                    )
                )
        detections = self._scale_detections(detections, scale)
        return detections, binary, scale

    def compute_hough_cloth_mask(
        self,
        image: np.ndarray,
        profile: ProductProfile,
        *,
        brightness_threshold: int = 120,
        downscale_factor: float = 1.0,
    ) -> np.ndarray:
        """Read-only cloth-vs-metal mask for the "Show cloth detection mask"
        diagnostic when detection.method="hough" (FIX 1). Shows exactly the
        region Hough is allowed to see — useful for tuning
        ``cloth_brightness_threshold``. Mirrors :meth:`detect_hough`'s
        internal mask computation exactly (same wide-kernel basis sized off
        ``profile``'s expected piece radius), so this preview never disagrees
        with what detection actually used. Does not affect detection results.
        """
        if image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        resized, scale = self._resize_for_detect(image)
        extract_profile = self._scaled_profile(profile, scale)
        gray = self._to_gray(resized)
        blurred = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)
        extra_scale = 1.0
        if downscale_factor > 1.0:
            extra_scale = 1.0 / downscale_factor
            new_w = max(1, int(blurred.shape[1] * extra_scale))
            new_h = max(1, int(blurred.shape[0] * extra_scale))
            blurred = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)
        expected_radius = (
            extract_profile.expected_width_px + extract_profile.expected_height_px
        ) / 4.0 * extra_scale
        basis = self._cloth_basis(blurred, expected_radius)
        return self._cloth_mask_from_basis(basis, brightness_threshold)

    def compute_cloth_region_mask(
        self,
        image: np.ndarray,
        *,
        brightness_threshold: int = 120,
    ) -> np.ndarray:
        """One-time cloth-vs-metal reference mask for Hough's cloth gating
        (detection.method="hough", see :meth:`detect_hough`).

        Unlike :meth:`compute_hough_cloth_mask` — the per-frame diagnostic that
        must survive dark dough sitting *on* the cloth (hence the
        morphological-close ``_cloth_basis``) — this is captured ONCE on the
        EMPTY cloth during setup (see ``MainWindow._on_save_cloth_reference``),
        so a plain brightness split is enough: bright Gärtuch (>= threshold) is
        cloth, the darker metal grating/mesh/frame is not. Returned at the
        input resolution (full cloth ROI, pre-crop, 0/255) so it can be stored
        and reloaded as-is, matching the captured frame's geometry — the same
        lifecycle as the bg_subtract reference.
        """
        if image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        gray = self._to_gray(image)
        return self._cloth_mask_from_basis(gray, brightness_threshold)

    @staticmethod
    def _cloth_basis(gray: np.ndarray, expected_radius: float) -> np.ndarray:
        """Background-brightness estimate used to tell bright cloth apart
        from dark metal, robust to the dough's own darkness.

        A plain per-pixel brightness threshold would blank out the dough
        circles themselves (they're dark, like the metal). A box-blur with a
        kernel only modestly larger than the piece still leaves the dough's
        own darkness dominating the local average (a blur kernel just larger
        than the piece's radius is NOT larger than its area). Morphological
        CLOSE with a kernel comfortably larger than the piece's diameter
        (~4x the expected radius, i.e. ~2x the diameter) instead genuinely
        erases small dark blobs — recovering the true surrounding cloth
        level — while a genuinely large dark metal/grating region (much
        bigger than the kernel) is left untouched. A rectangular kernel
        keeps this O(1)-per-pixel regardless of size (OpenCV's van
        Herk/Gil-Werman algorithm), so it stays cheap even at this width.
        Also doubles as the FILL value for the masked-out region in
        :meth:`detect_hough` (instead of blanking to 0), so no artificial
        sharp edge is introduced at the cloth/metal seam.

        Known limitation: within roughly one kernel-width of the cloth/metal
        boundary, closing can "bridge" a thin strip of metal into the bright
        side (the same property that lets it erase the dough also fills
        anything narrower than the kernel, including a boundary-adjacent
        metal strip). Use the "Show cloth detection mask" preview to check
        this against your real crop; the kernel was kept as small as
        possible (still ≥ one dough diameter) to minimize that band.
        """
        k = max(15, int(round(expected_radius * 4.0))) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _cloth_mask_from_basis(basis: np.ndarray, brightness_threshold: int) -> np.ndarray:
        _, mask = cv2.threshold(basis, brightness_threshold, 255, cv2.THRESH_BINARY)
        return mask

    def detect_bg_subtract(
        self,
        image: np.ndarray,
        reference: np.ndarray | None,
        profile: ProductProfile,
        *,
        threshold: int = 30,
    ) -> tuple[list[Detection], np.ndarray, float]:
        """ALTERNATIVE METHOD (detection.method="bg_subtract").

        Detects dough as the pixel difference from a captured empty-cloth
        ``reference`` frame ("whatever changed is dough") — independent of
        absolute brightness/threshold, so a shiny highlight that would punch
        a fake hole in an Otsu threshold has no effect here at all.

        ``reference`` must be the same shape as ``image`` (same crop, same
        resolution); if it's ``None`` or mismatched (e.g. no reference
        captured yet, or the cloth crop changed since capture), this returns
        no detections — callers should surface that via a throttled log/UI
        status rather than treat it as a crash.

        Returns ``(detections, binary, scale)``; ``binary`` is the ordinary
        Otsu/adaptive threshold (tripwire-compatible), unrelated to the diff
        mask used for extraction.
        """
        if image.size == 0:
            return [], np.zeros((1, 1), dtype=np.uint8), 1.0
        resized, scale = self._resize_for_detect(image)
        binary = self._threshold(
            cv2.GaussianBlur(self._to_gray(resized), (self._blur_ksize, self._blur_ksize), 0),
            invert=profile.dough_is_darker,
        )
        if reference is None or reference.shape[:2] != image.shape[:2]:
            return [], binary, scale

        extract_profile = self._scaled_profile(profile, scale)
        opened = self._bg_subtract_mask(resized, reference, threshold)
        detections = self._extract_external_only(opened, extract_profile, min_circularity=0.0)
        detections = self._scale_detections(detections, scale)
        return detections, binary, scale

    def compute_bg_subtract_mask(
        self,
        image: np.ndarray,
        reference: np.ndarray | None,
        *,
        threshold: int = 30,
    ) -> np.ndarray:
        """Read-only diff-threshold mask for the "Show cloth detection mask"
        diagnostic when detection.method="bg_subtract". Does not affect
        detection results — see :meth:`detect_bg_subtract`.
        """
        if image.size == 0 or reference is None or reference.shape[:2] != image.shape[:2]:
            return np.zeros((1, 1), dtype=np.uint8)
        resized, _scale = self._resize_for_detect(image)
        return self._bg_subtract_mask(resized, reference, threshold)

    def _bg_subtract_mask(
        self, resized_image: np.ndarray, reference: np.ndarray, threshold: int
    ) -> np.ndarray:
        resized_ref, _ = self._resize_for_detect(reference)
        gray_img = self._to_gray(resized_image)
        gray_ref = self._to_gray(resized_ref)
        blurred_img = cv2.GaussianBlur(gray_img, (self._blur_ksize, self._blur_ksize), 0)
        blurred_ref = cv2.GaussianBlur(gray_ref, (self._blur_ksize, self._blur_ksize), 0)
        diff = cv2.absdiff(blurred_img, blurred_ref)
        _, diff_binary = cv2.threshold(diff, max(1, threshold), 255, cv2.THRESH_BINARY)
        return self._morph_open(diff_binary)

    # ---------- internals ----------

    def _resize_for_detect(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Shared downscale step used by every detection method (see
        ``max_detect_width``). Returns ``(possibly-resized image, scale)``.
        """
        h, w = image.shape[:2]
        if not self._max_detect_width or w <= self._max_detect_width:
            return image, 1.0
        scale = self._max_detect_width / w
        new_h = max(1, int(h * scale))
        resized = cv2.resize(
            image, (self._max_detect_width, new_h), interpolation=cv2.INTER_AREA
        )
        return resized, scale

    @staticmethod
    def _scaled_profile(profile: ProductProfile, scale: float) -> ProductProfile:
        """Scale profile geometry fields to match a downscaled detect image."""
        if scale >= 1.0:
            return profile
        scale2 = scale * scale
        return profile.model_copy(
            update={
                "expected_area_px": max(1, int(profile.expected_area_px * scale2)),
                "expected_width_px": max(1, int(profile.expected_width_px * scale)),
                "expected_height_px": max(1, int(profile.expected_height_px * scale)),
            }
        )

    @staticmethod
    def _scale_detections(detections: list[Detection], scale: float) -> list[Detection]:
        """Scale detections back to original (full-resolution) coordinates."""
        if scale >= 1.0:
            return detections
        inv = 1.0 / scale
        return [
            Detection(
                bbox=(
                    int(d.bbox[0] * inv),
                    int(d.bbox[1] * inv),
                    int(d.bbox[2] * inv),
                    int(d.bbox[3] * inv),
                ),
                centroid=(d.centroid[0] * inv, d.centroid[1] * inv),
                area_px=int(d.area_px * inv * inv),
                width_px=int(d.width_px * inv),
                height_px=int(d.height_px * inv),
                confidence=d.confidence,
                label=d.label,
                circularity=d.circularity,
                solidity=d.solidity,
            )
            for d in detections
        ]

    def _extract_external_only(
        self,
        binary: np.ndarray,
        profile: ProductProfile,
        *,
        min_circularity: float = 0.0,
    ) -> list[Detection]:
        """Area + circularity gated extraction — NO solidity check.

        Shared by ``detect_contour_external`` and ``detect_bg_subtract``.
        Unlike ``_extract`` (the blob method), a hollow ring with an intact
        outer rim is never penalized: its outer-contour solidity would be
        near 1.0 anyway (the hole is invisible to RETR_EXTERNAL), but this
        skips even checking it, so callers don't need to fight the blob
        method's learned-reference solidity gate to get ring-tolerant
        behavior.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = profile.noise_threshold * profile.expected_area_px
        max_area = binary.shape[0] * binary.shape[1] * self._max_roi_fraction
        fused_area_gate = profile.expected_area_px * profile.fused_threshold
        fused_ar_limit = profile.fused_threshold * 2.0

        out: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ar_blob = max(w, h) / max(min(w, h), 1)

            perimeter = cv2.arcLength(cnt, True)
            circularity = (
                4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0.0 else 0.0
            )
            is_fused_candidate = area >= fused_area_gate and ar_blob <= fused_ar_limit
            if min_circularity > 0.0 and not is_fused_candidate and circularity < min_circularity:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2.0
                cy = y + h / 2.0
            label = self._classify(int(w), int(h), profile)
            confidence = float(min(1.0, area / max(profile.expected_area_px, 1)))
            out.append(
                Detection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    centroid=(float(cx), float(cy)),
                    area_px=int(area),
                    width_px=int(w),
                    height_px=int(h),
                    confidence=confidence,
                    label=label,
                    circularity=circularity,
                    solidity=solidity,
                )
            )
        return out

    @staticmethod
    def _fill_holes(binary: np.ndarray, kernel_size: int) -> np.ndarray:
        """EXPERIMENTAL (FILL_MASK_HOLES, ring/hollow-dough detection).

        A genuinely solid dough piece can threshold as a hollow ring when a
        specular highlight on its dome punches a "hole" through to
        background. Two steps:
          1. Morphological close with *kernel_size* bridges small breaks where
             the highlight reaches/notches the outer boundary.
          2. Re-drawing each external contour filled solid closes any
             still-enclosed hole, regardless of its shape.
        """
        if kernel_size > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        else:
            closed = binary
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = closed.copy()
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"unsupported image shape: {image.shape}")

    def _threshold(
        self,
        gray: np.ndarray,
        invert: bool = False,
        force_adaptive_block: int = 0,
    ) -> np.ndarray:
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        if invert:
            # Dark-on-bright mode (cloth).
            # Shadow-mask deep-dark machine-frame pixels so Otsu splits
            # {gray donuts} vs {white cloth} rather than {frame} vs {rest}.
            shadow_mask = gray < self._shadow_floor
            masked = gray.copy()
            masked[shadow_mask] = 255

            non_shadow = ~shadow_mask
            std = float(np.std(masked[non_shadow])) if non_shadow.any() else 0.0
            if std < 3.0:
                c = -self._adaptive_C
                result = cv2.adaptiveThreshold(
                    masked, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, self._adaptive_block | 1, c,
                )
            else:
                _, result = cv2.threshold(
                    masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            result[shadow_mask] = 0
            return result

        else:
            # Bright-on-dark mode (belt).
            # When a forced adaptive block is provided, skip global Otsu and
            # use local adaptive threshold directly.  This prevents nearby
            # donuts on an unevenly-lit belt from merging into a single blob:
            # the local mean adapts to each neighbourhood so the belt surface
            # between donuts is always classified as background.
            if force_adaptive_block > 1:
                return cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, force_adaptive_block | 1,
                    self._adaptive_C,
                )
            _, otsu_bin = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            std = float(np.std(gray))
            if std < 3.0:
                return cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, self._adaptive_block | 1,
                    self._adaptive_C,
                )
            return otsu_bin

    def _morph_open(self, binary: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._morph_ksize, self._morph_ksize)
        )
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    def _extract(
        self,
        binary: np.ndarray,
        profile: ProductProfile,
        *,
        min_solidity: float = 0.0,
        max_aspect_ratio: float = 0.0,
        min_circularity_shape: float = 0.0,
        min_solidity_shape: float = 0.0,
    ) -> list[Detection]:
        # RETR_EXTERNAL: only outermost contours — inner hole contours of donut
        # shapes are suppressed, so each physical ring counts as one piece.
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = profile.noise_threshold * profile.expected_area_px
        # Upper cap: reject blobs bigger than max_roi_fraction of the total image
        # area.  Machine-frame / gap blobs are orders of magnitude larger than
        # any individual dough piece, so this reliably excludes them.
        max_area = binary.shape[0] * binary.shape[1] * self._max_roi_fraction
        # A fused pair of two touching circles has area ~2× expected and
        # aspect ratio ~2:1.  A metal strip has much higher AR.  We use BOTH
        # conditions so metal strips are never exempt from the circularity gate.
        fused_area_gate = profile.expected_area_px * profile.fused_threshold
        fused_ar_limit = profile.fused_threshold * 2.0  # two circles side-by-side

        out: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            # Belt pre-filter: aspect ratio gate.
            # Elongated reflection streaks have a high long/short side ratio.
            # 0.0 = gate disabled.
            if max_aspect_ratio > 0.0:
                ar = max(w, h) / max(min(w, h), 1)
                if ar > max_aspect_ratio:
                    continue

            # Compute solidity (blob_area / convex_hull_area) once for both gates.
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            # Belt pre-filter: solidity gate.
            # Irregular grating reflections score low; real dough is compact.
            # 0.0 = gate disabled.
            if min_solidity > 0.0 and solidity < min_solidity:
                continue

            # Compute circularity (4π·area / perimeter²; 1.0 = perfect circle).
            perimeter = cv2.arcLength(cnt, True)
            circularity = (
                4.0 * math.pi * area / (perimeter * perimeter)
                if perimeter > 0.0
                else 0.0
            )

            # Learned-reference shape filter: solidity.
            # Applies to all blobs (singles and fused pairs are both solid;
            # metal reflections are irregular).
            if min_solidity_shape > 0.0 and solidity < min_solidity_shape:
                continue

            # Learned-reference shape filter: circularity.
            # Skipped for fused-pair candidates (two circles side-by-side have
            # lower circularity than a single circle).  A fused pair is large
            # AND has a moderate aspect ratio (≤ 2× diameter).  Metal strips are
            # also large but have extreme AR, so they must NOT be exempt.
            if min_circularity_shape > 0.0:
                ar_blob = max(w, h) / max(min(w, h), 1)
                is_fused_candidate = (
                    area >= fused_area_gate and ar_blob <= fused_ar_limit
                )
                if not is_fused_candidate and circularity < min_circularity_shape:
                    continue

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2.0
                cy = y + h / 2.0
            label = self._classify(int(w), int(h), profile)
            confidence = float(min(1.0, area / max(profile.expected_area_px, 1)))
            out.append(
                Detection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    centroid=(float(cx), float(cy)),
                    area_px=int(area),
                    width_px=int(w),
                    height_px=int(h),
                    confidence=confidence,
                    label=label,
                    circularity=circularity,
                    solidity=solidity,
                )
            )
        return out

    @staticmethod
    def _classify(width_px: int, height_px: int, profile: ProductProfile) -> str:
        w_ratio = width_px / max(profile.expected_width_px, 1)
        h_ratio = height_px / max(profile.expected_height_px, 1)

        # Sanity cap: reject blobs that are unrealistically large.  These are
        # almost always detection artifacts (merged blobs, machine frame residue)
        # rather than real defects.  Classifying them as "fused" would be
        # misleading; "unknown" still triggers FaultActive.
        if w_ratio > profile.unknown_max_ratio or h_ratio > profile.unknown_max_ratio:
            return "unknown"

        # wide = two donuts stuck horizontally (side-by-side) → fault.
        # tall = two donuts stuck vertically (one above other) → informational only.
        wide = w_ratio >= profile.fused_threshold
        tall = h_ratio >= profile.fused_threshold

        if wide and tall:
            return "unknown"
        if wide:
            return "row_fused"     # horizontal sticking → RED → FAULT
        if tall:
            return "column_fused"  # vertical stacking → ORANGE → no fault

        # Below single_min_ratio in either dimension → too small to be a real
        # donut; treat as noise / debris (unknown keeps it from silently passing).
        if w_ratio < profile.single_min_ratio or h_ratio < profile.single_min_ratio:
            return "unknown"

        return "single"
