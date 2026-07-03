"""Application configuration loader + saver.

Loads ``config/default.yaml`` first, then deep-merges ``config/local.yaml`` on
top if it exists. Saving writes the merged result back to ``local.yaml`` so
defaults remain a clean, version-controlled template.

On first run the SERVICE PIN hash is empty; :meth:`AppConfig.ensure_initialized`
hashes the literal ``"0000"`` and persists it so subsequent runs verify against
that hash instead of a plaintext fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from viscontrol.core.profiles import ProductProfile, ProfileStore
from viscontrol.core.security import hash_pin

Mode = Literal["demo", "production"]
Language = Literal["en", "de"]
CameraSource = Literal["auto", "basler", "mock"]
DetectionMethod = Literal["blob", "contour_external", "hough", "bg_subtract"]


class _AppSection(BaseModel):
    mode: Mode = "demo"
    language: Language = "en"
    active_profile: str = "Default"


class _CameraSection(BaseModel):
    source: CameraSource = "auto"
    serial: str = ""
    pixel_format: str = "Mono8"


class _MockCameraSection(BaseModel):
    image_dir: str = "assets/test_images"
    fps: float = Field(5.0, gt=0)


class _CaptureSection(BaseModel):
    """Raw-frame recording for offline calibration (see io/recorder.py)."""

    output_dir: str = "captures"


class _PlaybackSection(BaseModel):
    """Recorded-frame playback source for offline calibration (see
    io/camera.py PlaybackCamera). Switching to playback is a runtime-only
    toggle (SERVICE view); only these preferences persist across restarts.
    """

    folder: str = ""
    fps: float = Field(5.0, gt=0)
    loop: bool = True


class _OrientationSection(BaseModel):
    rotation: Literal[0, 90, 180, 270] = 0
    flip_horizontal: bool = False
    row_direction: Literal["vertical", "horizontal"] = "vertical"


class _InspectionSection(BaseModel):
    delay_after_pull_ms: int = Field(200, ge=0)
    fault_clear_frames: int = Field(5, ge=1)
    empty_cloth_threshold: int = Field(3, ge=0)
    empty_cloth_seconds: int = Field(30, ge=1)
    inspection_overlay_hold_seconds: float = Field(3.0, ge=0.0)
    belt_fault_debounce_frames: int = Field(3, ge=1)
    belt_fault_clear_frames: int = Field(3, ge=1)
    tripwire_check_interval: int = Field(1, ge=1)
    belt_check_interval: int = Field(1, ge=1)
    display_detection_interval: int = Field(1, ge=1)
    line_clear_debounce_frames: int = Field(3, ge=1)
    transfer_timeout_ms: int = Field(5000, ge=100)
    belt_detection_enabled: bool = Field(
        True,
        description=(
            "Debug toggle: when False, the belt inspection window never opens, "
            "no belt-side detection or fault debounce runs, and belt processing "
            "is skipped entirely in the frame loop. Cloth ROI detection and row-"
            "stop logic are unaffected. Runtime-only — see MainWindow's "
            "'Belt detection' checkbox (Service > Diagnostics)."
        ),
    )
    unknown_is_fault: bool = Field(
        False,
        description=(
            "FIX 6: when False (default), a blob the classifier couldn't "
            "confidently label single/fused is run through row/column "
            "orientation logic instead of an immediate hard fault — only a "
            "row-wise (horizontal) read raises FAULT_ROW_FUSED; a column-wise "
            "or genuinely ambiguous read is informational only "
            "(INFO_UNKNOWN_NONFAULT), never a hard fault or ext_error write. "
            "Set True to restore the old behavior (every 'unknown' blob is an "
            "immediate FAULT_UNKNOWN)."
        ),
    )


class _ContourExternalSection(BaseModel):
    """Params for DETECTION_METHOD="contour_external" (see detection/classical.py)."""

    min_circularity: float = Field(0.55, ge=0.0, le=1.0)


class _HoughSection(BaseModel):
    """Params for DETECTION_METHOD="hough" (see ClassicalDetector.detect_hough).

    ``min_dist_px``/``min_radius_px``/``max_radius_px`` of 0 mean "auto",
    derived from the active profile's expected piece diameter (tightened to
    0.8x-1.2x of the expected radius, vs the original 0.6x-1.3x — see FIX 2).
    """

    dp: float = Field(1.2, gt=0.0)
    min_dist_px: int = Field(0, ge=0)
    param1: float = Field(80.0, gt=0.0)
    param2: float = Field(
        45.0, gt=0.0,
        description=(
            "FIX 2: accumulator vote threshold — raised from 30.0 so weak/"
            "partial circle patterns (mesh texture, reflections) are rejected; "
            "only strong, complete, dough-sized circles pass."
        ),
    )
    min_radius_px: int = Field(0, ge=0)
    max_radius_px: int = Field(0, ge=0)
    radius_tolerance: float = Field(
        0.2, gt=0.0, lt=1.0,
        description=(
            "SECTION 2: half-width of the auto radius acceptance band as a "
            "fraction of the expected piece radius. The Hough min/max radius "
            "auto-derive to (1 - tol) … (1 + tol) × expected radius (was a "
            "hardcoded 0.8…1.2, i.e. tol=0.2). Lower = stricter dough-size "
            "gate. Only used when min_radius_px / max_radius_px are 0 (auto)."
        ),
    )
    gate_to_cloth: bool = Field(
        True,
        description=(
            "FIX 1 (HOUGH_GATE_TO_CLOTH): dough only ever sits on the bright "
            "cloth, never on the dark metal grating/mesh. When True, the "
            "non-cloth region (see cloth_brightness_threshold) is blanked "
            "before Hough runs (FIX 3 — grating/mesh texture is extremely "
            "edge-dense, so this is also the main speed win) AND any circle "
            "whose center still falls outside the cloth mask is rejected."
        ),
    )
    cloth_brightness_threshold: int = Field(
        120, ge=0, le=255,
        description=(
            "FIX 1: simple brightness split between the bright cloth (Gärtuch) "
            "and the darker metal grating/mesh/frame. Pixels >= this value are "
            "'cloth'; below are 'not cloth' and excluded from Hough entirely."
        ),
    )
    downscale_factor: float = Field(
        2.0, ge=1.0,
        description=(
            "SECTION 1 (was FIX 3): extra downscale applied ONLY for the Hough "
            "step, on top of the shared max_detect_width downscale. 1.0 = "
            "disabled. Circle coordinates/radii are scaled back up afterward. "
            "Default 2.0 — Hough's Canny+accumulator cost scales with pixel "
            "count, so halving each axis is ~4x faster per frame and is the "
            "main lever for keeping detection at camera rate (no frame drops)."
        ),
    )
    hough_interval_ms: int = Field(
        100, ge=10,
        description=(
            "FIX 1: maximum rate at which Hough detection runs, as minimum "
            "milliseconds between consecutive runs. 100 ms ≈ 10 detections/s. "
            "Hough only runs when TuchabzugRunning is True; between runs, the "
            "last known detection results are reused for display and row-state "
            "remains unchanged. If belt inspection is also due on the same "
            "frame, Hough is deferred by one frame (stagger) unless it hasn't "
            "run in >2× this interval. Lower = more responsive, higher CPU; "
            "higher = fewer frame drops."
        ),
    )
    cloth_reference_path: str = Field(
        "",
        description=(
            "Set automatically by the wizard's Calibration step (\"Save Cloth "
            "Reference\", see MainWindow._on_save_cloth_reference). Path to a "
            "one-time bright-cloth-vs-dark-metal mask captured on the empty "
            "cloth at cloth_brightness_threshold; empty means none saved yet. "
            "The cloth/camera don't move during operation, so it is captured "
            "once and reused rather than recomputed every frame."
        ),
    )


class _ShapeFilterSection(BaseModel):
    """SECTION 2: reject non-dough candidates by SHAPE and SIZE, not just
    brightness. Applied to the cloth-ROI detections of the active method
    (see InspectionPipeline.filter_dough_candidates) before they drive any
    stop decision, so reflections on metal (which are bright but not round
    and/or not dough-sized) no longer fire false stops.

    Size is gated relative to the LEARNED / detected dough diameter (no fixed
    pixels), so it adapts automatically to the dough size set by Learn
    Reference.
    """

    enabled: bool = Field(
        True,
        description=(
            "Master on/off for the shape+size rejection gate. When off, the "
            "active method's raw detections are used directly (legacy)."
        ),
    )
    min_circularity: float = Field(
        0.6, ge=0.0, le=1.0,
        description=(
            "A candidate must be at least this round (4*pi*area/perimeter^2) "
            "to count as dough. Hough already returns ideal circles "
            "(circularity 1.0 — its param2 accumulator is its own roundness "
            "gate), so this mainly tightens the blob/contour methods; "
            "reflections/streaks score low and are dropped."
        ),
    )
    size_tolerance: float = Field(
        0.4, gt=0.0, lt=1.0,
        description=(
            "Accept candidates whose mean diameter is within "
            "(1 ± size_tolerance) × the reference dough diameter (learned "
            "expected_width_px, falling back to the median detected "
            "diameter). Relative, not fixed pixels, so it scales with dough "
            "size. Lower = stricter."
        ),
    )


class _BandSection(BaseModel):
    """SECTION 3: active detection band near the transfer line.

    Only pieces whose LEADING edge falls inside this band (measured in
    cloth-ROI pixels relative to the transfer line) are considered for stop
    decisions. Pieces elsewhere in the ROI are still drawn but never drive a
    stop — this rejects far-away reflections/empty areas and cuts work.
    """

    enabled: bool = Field(
        True,
        description="Master on/off for the detection band. Off = whole ROI counts.",
    )
    ahead_px: int = Field(
        600, ge=0,
        description=(
            "How far the band extends to the APPROACH side of the transfer "
            "line (the cloth travels toward the line, so this is the side "
            "pieces arrive from). Cloth-ROI pixels."
        ),
    )
    behind_px: int = Field(
        150, ge=0,
        description=(
            "How far the band extends PAST the transfer line (the side a "
            "piece moves onto after crossing). Cloth-ROI pixels."
        ),
    )


class _BgSubtractSection(BaseModel):
    """Params for DETECTION_METHOD="bg_subtract".

    ``reference_path`` is set automatically by the "Capture empty-cloth
    reference" action in SERVICE > Detection; empty means no reference has
    been captured yet (the method then detects nothing, see classical.py).
    """

    threshold: int = Field(30, ge=1, le=255)
    reference_path: str = ""


class _DetectionSection(BaseModel):
    """Cloth-side piece-detection method selection.

    Belt detection is unaffected — it always uses the classical blob method
    (see InspectionPipeline.run_belt_inspection). Only cloth tracking
    (run_cloth_tracking) switches behavior based on ``method``, so all four
    methods can be A/B compared on the same scene without code changes.
    """

    method: DetectionMethod = "blob"
    use_row_grouping: bool = Field(
        False,
        description=(
            "Cloth side only. False (default) = legacy single-line tripwire "
            "(MainWindow._apply_tripwire_edge), behaviour unchanged. True = "
            "group detected pieces into rows by travel position and fire "
            "StopTuchabzug once per row as each row-line crosses the transfer "
            "line, so two back-to-back rows with no gap stop twice. See "
            "viscontrol/detection/row_grouping.py."
        ),
    )
    row_grouping_gap_diameters: float = Field(
        0.6,
        gt=0,
        description=(
            "USE_ROW_GROUPING only. Row split tolerance as a multiple of the "
            "median DETECTED piece diameter: pieces whose travel coordinates "
            "differ by less than (this × diameter) are the same row; a larger "
            "gap starts the next row. Scales with dough size. Lower = split more "
            "readily (more rows); higher = merge more (fewer rows). Typical "
            "0.5-1.0."
        ),
    )
    fill_mask_holes: bool = Field(
        False,
        description=(
            "blob-only: fill enclosed mask holes (e.g. a shiny dome's fake "
            "hollow ring) before blob extraction. Ignored by the other three "
            "methods, which don't depend on solid-blob fill."
        ),
    )
    fill_mask_holes_kernel: int = Field(9, ge=1)
    grid_columns: int = Field(
        8, ge=1,
        description=(
            "SECTION 6: number of dough pieces per row in the loaded grid. "
            "USE_ROW_GROUPING uses this to identify the FRONT/CURRENT row as "
            "the N pieces closest to the transfer bridge (by leading edge) and "
            "to fire one StopTuchabzug per row. Settable in the wizard."
        ),
    )
    row_gap_threshold_px: int = Field(
        150, ge=1,
        description=(
            "USE_ROW_GROUPING only. Pieces are sorted by leading-edge X and "
            "split into clusters wherever the gap to the next piece exceeds "
            "this many pixels — each cluster is one physical row, regardless "
            "of missing/extra detections. Replaces the old fixed-size "
            "sort-slice-by-grid_columns grouping. See "
            "viscontrol/detection/row_grouping.py: group_by_gap."
        ),
    )
    detection_zone_width_px: int = Field(
        600, ge=0,
        description=(
            "Width of the active Hough detection zone on the approach side of "
            "the transfer bridge, in cloth-ROI pixels. The detection region is "
            "[transfer_line - bridge_half, transfer_line + bridge_half + "
            "detection_zone_width_px]. Pixels outside this region are zeroed "
            "before the detection method runs so Hough never processes "
            "far-away cloth areas or reflections. 0 = disabled (full ROI). "
            "Default 600 px ≈ 2× typical dough diameter."
        ),
    )
    grid_rows: int = Field(
        0, ge=0,
        description=(
            "SECTION 6: expected number of rows on the cloth (0 = unknown / "
            "unbounded). Informational; row firing is driven per-row as each "
            "row reaches the bridge, not by this count."
        ),
    )
    max_memory_frames: int = Field(
        5, ge=0,
        description=(
            "USE_ROW_GROUPING only. Number of frames to hold the last known "
            "front-row leading-edge position when detection drops or returns "
            "fewer pieces than expected (frame drop fallback). 0 = disabled. "
            "See MainWindow._apply_tangent_stop_edge."
        ),
    )
    row_x_tolerance_px: int = Field(
        0, ge=0,
        description=(
            "USE_ROW_GROUPING only. Tolerance in pixels for grouping staggered "
            "pieces into the same row by leading-edge X. 0 = auto (1× detected "
            "piece diameter). Pieces whose leading_edge_x differs from the "
            "previous piece by at most this value belong to the same row. Set "
            "to ~1 piece diameter to absorb stagger within a row while still "
            "splitting clearly separated rows. Used identically by the stop "
            "decision and the display coloring (single source of truth)."
        ),
    )
    boundary_safety_margin_px: int = Field(
        80, ge=0,
        description=(
            "USE_ROW_GROUPING only. Minimum distance (px) that committed_boundary_x "
            "must stay BEHIND transfer_x. When a row fires, the boundary is set to "
            "min(fired_tangent_x, transfer_x - boundary_safety_margin_px). This "
            "prevents the boundary from overlapping the fire zone of the next row "
            "when the previous row fired very close to the transfer line. "
            "~half a dough diameter (default 80px) ensures the next row's pieces "
            "are never excluded at the instant they would otherwise satisfy the "
            "tangent <= transfer_x fire condition. 0 = no safety cap."
        ),
    )
    boundary_offset_px: int = Field(
        25, ge=0,
        description=(
            "USE_ROW_GROUPING only. Added to the fired tangent when setting "
            "committed_boundary_x: boundary = fired_tangent_x + boundary_offset_px. "
            "0 = boundary sits exactly at the fired tangent. Default 25px (was "
            "12px, originally 30px) balances a narrow dead zone above transfer_x "
            "against grouping/outlier logic that now also guards against stragglers."
        ),
    )
    clearing_timeout_ms: int = Field(
        400, ge=0,
        description=(
            "USE_ROW_GROUPING only. Safety timeout for the post-fire CLEARING -> "
            "ARMED transition: if the leftovers-cleared streak hasn't been "
            "satisfied within this many milliseconds of the next rising edge, "
            "ARMED is forced anyway so the boundary filter can't block a "
            "legitimate fire indefinitely. Default 400ms (was a hardcoded 1.5s)."
        ),
    )
    post_reset_fresh_margin_px: int = Field(
        200, ge=0,
        description=(
            "USE_ROW_GROUPING only. After a FULL RESET (all rows complete, "
            "active_row_index back to 0), the first fire of the new cycle is "
            "gated: the active row's tangent must be seen ABOVE "
            "(transfer_x + this margin) at least once before it can fire. "
            "This prevents leftover/straggler pieces from the just-completed "
            "cycle from instant-firing as the new Row 1. Does NOT apply after "
            "a mid-cycle ADVANCE — Row 2+ can fire immediately. "
            "~1–2 dough diameters (default 200px) is recommended. "
            "0 = disabled (no post-reset gate)."
        ),
    )
    contour_external: _ContourExternalSection = Field(default_factory=_ContourExternalSection)
    hough: _HoughSection = Field(default_factory=_HoughSection)
    bg_subtract: _BgSubtractSection = Field(default_factory=_BgSubtractSection)
    shape_filter: _ShapeFilterSection = Field(default_factory=_ShapeFilterSection)
    band: _BandSection = Field(default_factory=_BandSection)


class _OpcuaSection(BaseModel):
    endpoint: str = "opc.tcp://0.0.0.0:4840/viscontrol/"
    namespace: str = "http://opelka.com/viscontrol"


class _WebSection(BaseModel):
    enabled: bool = True
    port: int = Field(8080, ge=1, le=65535)
    password_hash: str = ""


class _UiSection(BaseModel):
    service_pin_hash: str = ""
    recent_defects_max: int = Field(20, ge=1)
    startup_banner_seconds: int = Field(30, ge=0)


class _StorageSection(BaseModel):
    log_dir: str = "logs"
    defect_image_dir: str = "logs/defects"
    app_log_rotation_mb: int = Field(10, ge=1)
    app_log_keep_files: int = Field(7, ge=1)
    csv_log_keep_days: int = Field(90, ge=1)


class _PlcSection(BaseModel):
    """Production PLC OPC UA client settings (only used when mode='production')."""

    url: str = "opc.tcp://192.168.178.150:4840"
    node_ext_tuchabzug_stop: str = "ns=6;s=::TUA:fromext_stop_Tuchabzug"
    node_ext_tuchabzug_status: str = "ns=6;s=::TUA:toext_Tuchabzug_running"
    node_ext_error: str = "ns=6;s=::AsGlobalPV:fromext_Error_idx"
    node_ext_viscontrol_alive: str = "ns=6;s=::Signal:fromext_viscontrol_alive"
    node_ext_error_quit: str = "ns=6;s=::Signal:toext_Error_quit"
    node_ext_einlaufband_running: str = "ns=6;s=::Einlauf:toext_Einlaufband_running"
    poll_interval_s: float = Field(0.1, gt=0)
    stop_pulse_ms: int = Field(100, ge=10)
    fault_error_code: int = Field(2, ge=0)
    reconnect_delay_s: float = Field(2.0, gt=0)


class AppConfig(BaseModel):
    """Root config object. All other code reads/writes via this."""

    app: _AppSection = Field(default_factory=_AppSection)
    camera: _CameraSection = Field(default_factory=_CameraSection)
    mock_camera: _MockCameraSection = Field(default_factory=_MockCameraSection)
    capture: _CaptureSection = Field(default_factory=_CaptureSection)
    playback: _PlaybackSection = Field(default_factory=_PlaybackSection)
    orientation: _OrientationSection = Field(default_factory=_OrientationSection)
    inspection: _InspectionSection = Field(default_factory=_InspectionSection)
    detection: _DetectionSection = Field(default_factory=_DetectionSection)
    profiles: list[ProductProfile] = Field(default_factory=list)
    opcua: _OpcuaSection = Field(default_factory=_OpcuaSection)
    web: _WebSection = Field(default_factory=_WebSection)
    ui: _UiSection = Field(default_factory=_UiSection)
    storage: _StorageSection = Field(default_factory=_StorageSection)
    plc: _PlcSection = Field(default_factory=_PlcSection)

    @field_validator("profiles")
    @classmethod
    def _at_least_one_profile(cls, v: list[ProductProfile]) -> list[ProductProfile]:
        if not v:
            raise ValueError("at least one profile is required")
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError("profile names must be unique")
        return v

    def profile_store(self) -> ProfileStore:
        """Build a :class:`ProfileStore` from the current profile list."""
        return ProfileStore(self.profiles)

    def active_profile(self) -> ProductProfile:
        store = self.profile_store()
        if not store.has(self.app.active_profile):
            # Fall back to the first profile if the configured active one is gone.
            self.app.active_profile = store.names()[0]
        return store.get(self.app.active_profile)

    def ensure_initialized(self) -> bool:
        """Populate one-time runtime defaults. Returns True if anything changed.

        Currently this means hashing the default ``"0000"`` PIN if the stored
        hash is empty. Called on every load so a freshly cloned config still
        gets a usable PIN without leaving plaintext in YAML.
        """
        changed = False
        if not self.ui.service_pin_hash:
            self.ui.service_pin_hash = hash_pin("0000")
            changed = True
        return changed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — ``override`` wins for scalars and lists."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    config_dir: Path,
    *,
    local_filename: str = "local.yaml",
    default_filename: str = "default.yaml",
) -> AppConfig:
    """Load ``default.yaml`` + optional ``local.yaml`` and return an :class:`AppConfig`.

    Why deep-merge rather than full override: the user may only want to flip
    ``app.mode`` to production without copy-pasting the entire defaults file
    and risking drift when defaults change in a future release.
    """
    config_dir = Path(config_dir)
    default_path = config_dir / default_filename
    if not default_path.exists():
        raise FileNotFoundError(f"defaults not found: {default_path}")

    with default_path.open("r", encoding="utf-8") as f:
        merged: dict[str, Any] = yaml.safe_load(f) or {}

    local_path = config_dir / local_filename
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, local)

    cfg = AppConfig.model_validate(merged)
    cfg.ensure_initialized()
    return cfg


def save_config(
    cfg: AppConfig,
    config_dir: Path,
    *,
    local_filename: str = "local.yaml",
) -> Path:
    """Atomically write the full config to ``local.yaml`` and return the path."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / local_filename
    tmp = target.with_suffix(target.suffix + ".tmp")
    data = cfg.model_dump(mode="json")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    tmp.replace(target)
    return target
