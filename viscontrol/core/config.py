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
    fill_mask_holes: bool = Field(
        False,
        description=(
            "blob-only: fill enclosed mask holes (e.g. a shiny dome's fake "
            "hollow ring) before blob extraction. Ignored by the other three "
            "methods, which don't depend on solid-blob fill."
        ),
    )
    fill_mask_holes_kernel: int = Field(9, ge=1)
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
    max_memory_frames: int = Field(
        5, ge=0,
        description=(
            "Number of frames a PENDING tracked cluster may go unmatched "
            "(e.g. a dropped Hough frame) before it's dropped from tracking. "
            "0 = disabled (drop immediately on any miss). See "
            "viscontrol/detection/proximity_clustering.py: ClusterTracker."
        ),
    )
    cycle_idle_reset_ms: int = Field(
        3000, ge=1,
        description=(
            "Logging/observability only — does not gate firing. When zero "
            "pieces have been detected for this many milliseconds, the "
            "current batch is considered finished: a CYCLE-SUMMARY line is "
            "logged (clusters fired this batch, their overshoot values, "
            "timing), then the batch-scoped counters, the DONE-cluster set, "
            "and the excluded-piece-identity set (DonePieceTracker) all "
            "reset for the next batch. Clustering/firing itself keeps "
            "running continuously regardless of this boundary."
        ),
    )
    min_fresh_confirmations: int = Field(
        2, ge=1,
        description=(
            "Quality gate (diagnostic Fix D): a candidate piece must appear "
            "in this many consecutive FRESH Hough frames (tangent_x-matched "
            "within proximity_clustering.tolerance_px) before it becomes "
            "eligible for clustering/firing — filters out one-frame "
            "reflections/noise. 1 disables the gate entirely: every piece "
            "passes through unchanged, zero overhead. See "
            "viscontrol/detection/proximity_clustering.py: "
            "PieceConfirmationTracker."
        ),
    )
    roi_valid_y_min: float | None = Field(
        None,
        description=(
            "Y-axis sanity gate (diagnostic Fix E): pieces with center_y "
            "below this value are rejected before clustering/confirmation. "
            "null = disabled. Y is never used for matching or clustering "
            "itself — this is a pure presence filter for impossible "
            "detections (e.g. a reflection above/below the cloth)."
        ),
    )
    roi_valid_y_max: float | None = Field(
        None,
        description="Y-axis sanity gate upper bound (diagnostic Fix E). null = disabled.",
    )
    contour_external: _ContourExternalSection = Field(default_factory=_ContourExternalSection)
    hough: _HoughSection = Field(default_factory=_HoughSection)
    bg_subtract: _BgSubtractSection = Field(default_factory=_BgSubtractSection)
    shape_filter: _ShapeFilterSection = Field(default_factory=_ShapeFilterSection)
    band: _BandSection = Field(default_factory=_BandSection)


class _ColumnLearningSection(BaseModel):
    """DIAGNOSTIC (logging-only): automatic column detection via Y-centroid
    learning. See viscontrol/detection/column_learning.py (ColumnLearner).

    Purely observational groundwork for a possible future column-based
    tracking mode — never read by the cluster-based fire decision (see
    viscontrol/detection/proximity_clustering.py: ClusterTracker) or any
    state-machine decision.
    """

    enabled: bool = Field(
        True,
        description=(
            "Master on/off for COLUMN-LEARN logging and column assignment. "
            "When False, no rolling window is fed and no COLUMN-LEARN lines "
            "are logged."
        ),
    )
    window_size: int = Field(
        100, ge=1,
        description="Rolling window size: number of recent piece Y-centroids retained.",
    )
    recompute_every_n_frames: int = Field(
        30, ge=1,
        description=(
            "How often (in frames fed to ColumnLearner.update) column_y_bands "
            "are recomputed from the current window, rather than every frame, "
            "so bands stay stable rather than jittery."
        ),
    )
    expected_columns: int = Field(
        8, ge=1,
        description=(
            "Number of Y-bands ColumnLearner splits its rolling window into. "
            "Decoupled from the retired detection.grid_columns (a different, "
            "row-firing-only field) — this is ColumnLearner's own knob."
        ),
    )


class _ProximityClusteringSection(BaseModel):
    """Tangent-based proximity clustering: cluster_by_tangent (see
    viscontrol/detection/proximity_clustering.py) groups detected pieces by
    proximity along tangent_x only (X axis), with no fixed expected count
    and no outlier rejection — this is the grid-free replacement for the
    old fixed-grid/row model.

    ``tolerance_px`` is now load-bearing for the actual StopTuchabzug fire
    decision (MainWindow._apply_cluster_stop_edge / ClusterTracker), not
    just the diagnostic overlay. ``enabled`` still only gates the
    PROXIMITY-CLUSTER diagnostic log line and overlay — firing itself runs
    unconditionally regardless of this flag.
    """

    enabled: bool = Field(
        True,
        description=(
            "Master on/off for PROXIMITY-CLUSTER logging and the cluster "
            "overlay only. When False, no diagnostic log/overlay is "
            "produced, but tolerance_px still governs live firing."
        ),
    )
    tolerance_px: int = Field(
        150, ge=1,
        description=(
            "Gap threshold along tangent_x: a new cluster starts whenever "
            "the tangent_x gap between two sorted pieces exceeds this many "
            "pixels. Also used as the cross-frame cluster-matching tolerance "
            "in ClusterTracker. Live-adjustable from Service > Diagnostics — "
            "takes effect on the next frame, no restart needed."
        ),
    )


class _TransferOrchestratorSection(BaseModel):
    """THE SINGLE SOURCE OF TRUTH for row/transfer state (see
    viscontrol/core/transfer_orchestrator.py: TransferOrchestrator). Owns
    row identity, the row lifecycle state machine, and the single call path
    to StopTuchabzug. Supersedes the retired ClusterTracker/DonePieceTracker
    fire path and the diagnostic TransferEventTracker.
    """

    enabled: bool = Field(
        True,
        description=(
            "Master on/off. When False, _run_transfer_orchestrator returns "
            "immediately and NO stop is ever issued (fail-safe: the machine "
            "simply does not stop — there is no fallback to the retired "
            "fire path)."
        ),
    )
    row_match_distance_px: float = Field(
        120, ge=0,
        description="Max interval gap between an observation's and a row's tangent_x range to count as a match.",
    )
    row_group_merge_px: float = Field(
        130, ge=0,
        description=(
            "Max interval distance between two unmatched upstream observations for them to be "
            "grouped into a single new row candidate (one physical multi-column row)."
        ),
    )
    row_new_min_upstream_px: float = Field(
        60, ge=0,
        description="An unmatched observation must have front_tangent > transfer_x + this to seed a new row.",
    )
    row_new_min_gap_px: float = Field(
        150, ge=0,
        description=(
            "Minimum interval-distance separation from every existing non-terminal row for an "
            "unmatched observation to become a new row candidate — prevents adjacent rows merging."
        ),
    )
    row_arm_min_frames: int = Field(
        3, ge=1,
        description="Minimum confirmed_upstream_frames before a DETECTED row can become ACTIVE.",
    )
    row_max_missed_frames: int = Field(
        8, ge=1,
        description="A DETECTED or ACTIVE row is ABANDONED after this many consecutive missed frames.",
    )
    row_exit_margin_px: float = Field(
        80, ge=0,
        description="A TRANSFERRING row becomes TRANSFERRED once front_tangent < transfer_x - this.",
    )
    stop_ack_timeout_s: float = Field(
        5.0, ge=0,
        description="A STOP_REQUESTED row is ABANDONED if the PLC never acknowledges within this many seconds.",
    )
    transfer_complete_timeout_s: float = Field(
        15.0, ge=0,
        description="Safety-net timeout: a TRANSFERRING row is forced to TRANSFERRED after this many seconds.",
    )


class _LayoutQualitySection(BaseModel):
    """Layout-quality safety lock (see viscontrol/core/layout_quality.py:
    LayoutQualityMonitor). Detects when cluster_by_tangent's row split is too
    ambiguous to trust — a within-row tangent_x gap larger than a
    between-row gap, which no tolerance_px value can separate — and stops
    the cloth + raises a fault instead of guessing. Never re-clusters, never
    changes cluster_by_tangent/Hough/tolerance_px; only measures the SAME
    cluster list the production path already computed.
    """

    enabled: bool = Field(
        True,
        description="Master on/off. When False, the monitor is never constructed/consulted.",
    )
    min_separation_ratio: float = Field(
        1.8, gt=0.0,
        description=(
            "The smallest between-row gap must be at least this many times "
            "the biggest within-row gap for the layout to be considered "
            "unambiguous. Measured from real logs: good loads 2.5 and 3.7, "
            "a hand-staggered bad load 0.79. 1.8 sits clearly between."
        ),
    )
    ambiguous_frames_to_trigger: int = Field(
        5, ge=1,
        description=(
            "Consecutive FRESH ambiguous measurements required before the "
            "lock triggers. Never trigger on a single noisy frame — this "
            "debounce is what stops the old row-splitting flicker from "
            "reappearing as a flickering stop."
        ),
    )
    min_pieces_for_check: int = Field(
        6, ge=1,
        description=(
            "Below this many total detected pieces, evidence is too thin "
            "to judge — a partly-visible cloth at the start of a batch is "
            "normal, not a fault."
        ),
    )
    min_within_gap_px: float = Field(
        5.0, ge=0.0,
        description=(
            "Divide-by-near-zero guard: a within-row gap below this is "
            "treated as unambiguous (ratio = +inf) rather than blown up "
            "into a meaningless huge ratio."
        ),
    )
    error_code: int = Field(
        2, ge=0,
        description=(
            "ext_error value written on a layout fault. NOTE: 2 currently "
            "also means 'stuck/fused row' (see plc.fault_error_code). The "
            "operator action differs (fused = remove stuck dough; layout = "
            "re-space it), so a distinct code is preferable once the PLC "
            "engineer can allocate one — changing this value is the only "
            "edit required."
        ),
    )
    auto_resume_on_good_layout: bool = Field(
        False,
        description=(
            "False (default, recommended): the lock clears ONLY on operator "
            "acknowledge (ext_error_quit) — the operator must look at the "
            "cloth before production resumes. True: additionally auto-clear "
            "after good_frames_to_resume consecutive good measurements; use "
            "only if operators find the acknowledge step burdensome."
        ),
    )
    good_frames_to_resume: int = Field(
        10, ge=1,
        description="Consecutive good measurements required to auto-clear. Only used when auto_resume_on_good_layout is True.",
    )


class _SizeCalibrationSection(BaseModel):
    """Development-only piece-size measurement tool (see
    viscontrol/core/size_calibration.py: SizeCalibrator). Setup-time only —
    run from the installation wizard's Calibration page to measure the
    actual dough piece radius (which can vary ~0.5x-2x between batches) and
    suggest corrected detection.hough radius limits / profile geometry.
    Never read by the production detection pipeline itself.
    """

    enabled: bool = Field(
        True,
        description="Master on/off for the wizard's Size Calibration section.",
    )
    capture_frames: int = Field(30, ge=1, description="Frames captured per Measure press.")
    sweep_min_radius_px: int = Field(
        20, ge=1,
        description="Wide-open Hough minRadius used ONLY for calibration measurement.",
    )
    sweep_max_radius_px: int = Field(
        400, ge=1,
        description="Wide-open Hough maxRadius used ONLY for calibration measurement.",
    )
    outlier_reject_frac: float = Field(
        0.35, gt=0.0, lt=1.0,
        description=(
            "Radii outside median * (1 +/- this) are discarded before "
            "recomputing median/p10/p90/stddev — robust to reflections and "
            "machine-part circles far from the true piece size."
        ),
    )
    radius_margin: float = Field(
        0.30, gt=0.0, lt=1.0,
        description=(
            "+/- fraction around the measured median radius used to derive "
            "suggested_min_radius_px / suggested_max_radius_px. Must absorb "
            "Hough's own radius jitter plus real piece-to-piece variation."
        ),
    )
    min_frames: int = Field(10, ge=1, description="Below this many captured frames, ok=False.")
    min_pieces: int = Field(
        4, ge=1, description="Below this many pieces/frame (median), ok=False."
    )
    max_radius_cv: float = Field(
        0.25, gt=0.0,
        description="stddev/median above this (post outlier-rejection) => ok=False, too_variable.",
    )
    max_plausible_pieces_per_frame: int = Field(
        40, ge=1,
        description=(
            "Pass-2 sanity cap (see SizeCalibrator.add_frame's two-pass "
            "duplicate-circle rejection): if a frame still returns more "
            "detections than this after minDist is scaled to the "
            "pass-1 rough radius estimate, the whole frame is discarded "
            "(not stored, not counted) rather than letting an implausible "
            "count corrupt the radius median."
        ),
    )


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
    column_learning: _ColumnLearningSection = Field(default_factory=_ColumnLearningSection)
    proximity_clustering: _ProximityClusteringSection = Field(default_factory=_ProximityClusteringSection)
    transfer_orchestrator: _TransferOrchestratorSection = Field(default_factory=_TransferOrchestratorSection)
    layout_quality: _LayoutQualitySection = Field(default_factory=_LayoutQualitySection)
    size_calibration: _SizeCalibrationSection = Field(default_factory=_SizeCalibrationSection)
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
