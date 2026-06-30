"""Main window — wires the sidebar, views, and runtime services together.

This is the "controller" layer. It owns:
- the AppConfig (passed in)
- the camera, detector, pipeline, state machine, event log
- the inference / capture / review / service / wizard views
- the OPC UA bridge (Production) and web server thread

Cross-thread frame / detection signals come in via Qt queued connections so
the UI thread only ever sees clean data.
"""


from __future__ import annotations

import copy
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from viscontrol.core.config import AppConfig, save_config
from viscontrol.core.event_log import Event, EventLog, EventType
from viscontrol.core.events import Mode, PipelineResult, RowPhase, State, Verdict
from viscontrol.core.logger import logger
from viscontrol.core.profiles import ProductProfile
from viscontrol.core.state_machine import StateMachine
from viscontrol.detection.calibration import learn_reference
from viscontrol.detection.classical import ClassicalDetector
from viscontrol.detection.pipeline import InspectionPipeline
from viscontrol.detection.row_grouping import (
    RowLineTracker,
    leading_edge_x,
    median_piece_diameter,
    slice_rows_by_count,
)
from viscontrol.io.camera import OrientationTransform, PlaybackCamera, make_camera
from viscontrol.io.recorder import FrameRecorder
from viscontrol.ui.i18n import install_translator
from viscontrol.ui.theme import (
    ACCENT_RED,
    BACKGROUND,
    FONT_SMALL,
    GLOBAL_STYLESHEET,
    SUCCESS_GREEN,
    TEXT_SECONDARY,
)
from viscontrol.ui.views.capture_view import CaptureView
from viscontrol.ui.views.inference_view import InferenceCounts, InferenceView
from viscontrol.ui.views.review_view import ReviewView
from viscontrol.ui.views.service_view import ServiceView
from viscontrol.ui.views.wizard_view import WizardView
from viscontrol.ui.widgets.sidebar import Sidebar
from viscontrol.ui.widgets.status_bar import StatusBar

# Display refresh is capped so that expensive QImage/QPixmap conversion in
# paintEvent does not run at the full camera acquisition rate.  Detection
# always runs at camera rate (subject to belt_check_interval / tripwire_check_interval).
_DISPLAY_INTERVAL_S: float = 1.0 / 15.0   # ~15 fps display cap
_PERF_LOG_INTERVAL_S: float = 1.0         # SECTION 1: per-stage timing summary, once/sec


class _FrameBridge(QObject):
    """Bridges background-thread frame callbacks into the Qt event loop.

    Camera callbacks run on the camera's worker thread; we re-emit via Qt
    signals so the slots in the main window always run on the GUI thread.
    """

    new_frame = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._processing: bool = False  # True while _on_frame is running on the GUI thread
        self._dropped: int = 0           # frames silently dropped due to backpressure

    def emit_frame(self, frame: np.ndarray) -> None:
        if self._processing:
            self._dropped += 1
            return
        self.new_frame.emit(frame)


class MainWindow(QMainWindow):
    """The application's single top-level window."""

    def __init__(
        self,
        config: AppConfig,
        config_dir: Path,
        *,
        web_server_factory: Optional[Callable] = None,
        opcua_server_factory: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self._cfg = config
        self._config_dir = config_dir
        self._web_server_factory = web_server_factory
        self._opcua_server_factory = opcua_server_factory

        self.setWindowTitle("VisControl — OPELKA")
        self.resize(1400, 900)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # Core services.
        self._detector = ClassicalDetector()
        self._pipeline = InspectionPipeline(self._detector)
        self._sm = StateMachine(fault_clear_frames=config.inspection.fault_clear_frames)
        self._event_log = EventLog(Path(config.storage.log_dir))
        self._counts = InferenceCounts()
        self._latest_frame: np.ndarray | None = None
        self._latest_annotated: np.ndarray | None = None
        self._latest_pipeline_ms = 0.0
        self._sticky_belt_detections: list = []
        self._sticky_belt_crop_rect: tuple | None = None
        self._sticky_overlay_timer = QTimer(self)
        self._sticky_overlay_timer.setSingleShot(True)
        self._sticky_overlay_timer.timeout.connect(self._clear_sticky_overlay)

        # Camera + bridge.
        self._frame_bridge = _FrameBridge()
        self._frame_bridge.new_frame.connect(self._on_frame)
        self._camera = make_camera(
            source=config.camera.source,
            mock_image_dir=Path(config.mock_camera.image_dir),
            mock_fps=config.mock_camera.fps,
            pixel_format=config.camera.pixel_format,
            basler_serial=config.camera.serial,
            transform=self._build_transform(),
            on_warning=self._on_camera_warning,
        )
        self._camera_running = False
        # CALIBRATION TOOLING: raw-frame recorder (see io/recorder.py) and the
        # live<->playback runtime toggle (see io/camera.py PlaybackCamera).
        # Neither has any effect on detection/tripwire/state-machine behavior.
        self._recorder = FrameRecorder(Path(config.capture.output_dir))
        self._playback_active = False
        self._recorder_status_timer = QTimer(self)
        self._recorder_status_timer.setInterval(250)
        self._recorder_status_timer.timeout.connect(self._refresh_recording_status)
        self._playback_status_timer = QTimer(self)
        self._playback_status_timer.setInterval(300)
        self._playback_status_timer.timeout.connect(self._refresh_playback_status)
        # Detection is enabled/disabled by the START/STOP button; the camera
        # itself always runs from launch so all views show live feed immediately.
        self._detection_enabled = False
        # Timestamp used to throttle web JPEG encoding (≤ 2 FPS).
        self._last_web_encode = 0.0
        # Timestamp of the last "dropped N frames" log (for 5-second throttling).
        self._dropped_frames_last_log: float = time.monotonic()
        # Timestamp of the last UI display refresh (throttled to _DISPLAY_INTERVAL_S).
        self._last_display_refresh: float = 0.0
        # SECTION 1: per-stage timing accumulators for the 1-s performance
        # summary (logged at INFO, throttled — so the bottleneck is visible).
        self._perf_totals: dict[str, float] = {
            k: 0.0 for k in ("split", "detect", "rowgroup", "belt", "display", "total")
        }
        self._perf_frames: int = 0
        self._perf_detect_count: int = 0   # detection passes in the current window
        self._perf_last_log: float = 0.0
        # Cached active profile so _on_frame avoids rebuilding ProfileStore every frame.
        self._active_profile_cache: "ProductProfile" = config.active_profile()
        # Last known frame dimensions — used to update coord captions on shape change.
        self._last_frame_w: int = 0
        # Tripwire edge-detection state.  Reset when a new TRACKING session starts.
        self._tripwire_prev_stable: bool = False   # last accepted (debounced) occupancy
        self._tripwire_candidate: bool = False     # candidate state being debounced
        self._tripwire_debounce_count: int = 0     # consecutive frames in candidate state
        # Belt fault debounce state for the concurrent belt check.
        self._belt_fault_debounce_count: int = 0   # consecutive fault-verdict frames
        self._belt_fault_armed: bool = True        # True = ready to raise a new fault
        self._belt_clear_count: int = 0            # consecutive clean frames toward re-arm
        # Frame counter for check-interval throttling.
        self._frame_count: int = 0
        # Previous tuchabzug_running value — detect rising edge in _on_frame.
        self._prev_tuchabzug_running: bool = False
        # Row-at-line state model: prevents StopTuchabzug re-firing during transfer.
        self._row_phase: RowPhase = RowPhase.IDLE
        self._line_clear_debounce_count: int = 0   # clear frames counted in TRANSFERRING
        self._transfer_start_time: float = 0.0     # time.monotonic() when TRANSFERRING began
        # Row grouping (USE_ROW_GROUPING) — only active when detection.use_row_grouping
        # is True; otherwise the legacy tripwire above runs unchanged. Reset per cycle.
        self._row_line_tracker = RowLineTracker()
        self._row_group_stop_active: bool = False  # mirrors StopTuchabzug pill for the pulse
        self._row_lines: list[float] = []          # current row-line x's (display)
        self._row_count: int = 0                   # current detected row count (display)
        # SECTION 5/6: leading-edge centroids of the CURRENT/front row (the
        # grid_columns pieces nearest the transfer bridge), drawn in a distinct
        # colour so the operator sees which pieces are grouped as one row.
        self._current_row_centroids: list[tuple[float, float]] = []
        # SECTION 3: active detection band (x_left, x_right) in cloth-local
        # coords, for display; None = whole ROI / band disabled.
        self._detection_band: tuple[int, int] | None = None
        # FIX 1: Hough rate-limiting state. _last_hough_time=0.0 means
        # "run immediately on next eligible frame" (set on session reset).
        self._last_hough_time: float = 0.0
        self._cached_cloth_detections: list = []
        self._cached_cloth_highlight: list[tuple[float, float]] = []
        self._cached_tripwire_occupied: bool = False
        # FIX 2: size-adaptive bridge/band width (px) computed from detected
        # piece diameter; 0 = not yet computed, fallback to profile value.
        self._active_bridge_width_px: int = 0
        # Detection zone outer boundary (approach-side, cloth-local x), for
        # display; None when zone disabled or detection hasn't run yet.
        self._detection_zone_outer_x: int | None = None
        # Grid-aware tangent stop (USE_ROW_GROUPING=ON): explicit per-row state machine.
        # _active_row_index advances on TuchabzugRunning rising edge once the active row
        # is STOPPED. All rows DONE triggers a full-cycle reset.
        self._active_row_index: int = 0
        self._row_status: list[str] = ["WAITING"] * max(0, config.detection.grid_rows)
        self._row_advance_pending_log: bool = False  # diagnostic only: set True when row advances
        self._committed_boundary_x: float | None = None  # leading_edge_x of last fired row; filters leftovers
        self._cycle_partial_warned: bool = False     # True once partial-detection warning logged
        self._cycle_front_row: list = []             # active-row detections (for display + stop)
        self._cycle_row2: list = []                  # next-row detections (for display)
        self._cycle_ref_tangent_x: float | None = None  # effective active-row leading tangent x
        self._cycle_log_next_time: float = 0.0      # throttle gate for per-cycle state log
        # FIX 2: position memory fallback (USE_ROW_GROUPING only).
        # Stores last known active-row leftmost tangent for up to max_memory_frames
        # frames when detection drops. Cleared on every TuchabzugRunning rising edge.
        self._last_known_row_leading_x: float | None = None
        self._memory_frame_count: int = 0
        # DIAGNOSTIC (temporary — two-rows-at-once investigation, see
        # InspectionPipeline.DIAGNOSTIC_ROW_PROFILE). Throttle gate for the
        # ~3 Hz bump/valley log; does not affect detection or tripwire state.
        self._diag_row_profile_last_log: float = 0.0
        self._transfer_timeout_warned: bool = False  # True once timeout warning was logged
        # Wizard draft: suppress file writes during wizard; revert on cancel.
        self._wizard_active: bool = False
        self._wizard_cfg_snapshot = None
        self._belt_mask_enabled: bool = False
        self._cloth_mask_enabled: bool = False
        # bg_subtract reference (full cloth ROI, pre-crop). Loaded from
        # config.detection.bg_subtract.reference_path at startup if set;
        # refreshed in-memory by _on_capture_bg_reference(). See
        # InspectionPipeline.run_cloth_tracking / ClassicalDetector.detect_bg_subtract.
        self._bg_reference_cloth: np.ndarray | None = None
        ref_path_str = config.detection.bg_subtract.reference_path
        if ref_path_str:
            ref_path = Path(ref_path_str)
            if ref_path.exists():
                loaded_ref = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
                if loaded_ref is not None:
                    self._bg_reference_cloth = loaded_ref
                else:
                    logger.warning("could not load bg_subtract reference: {}", ref_path)
        # Cloth Reference (full cloth ROI, pre-crop, binary mask 0/255).
        # Loaded from config.detection.hough.cloth_reference_path at startup
        # if set; refreshed in-memory by _on_save_cloth_reference(). See
        # InspectionPipeline.run_cloth_tracking / ClassicalDetector.detect_hough.
        # The cloth/camera don't move during operation, so this is captured
        # once during setup and reused as-is (same lifecycle as
        # _bg_reference_cloth above) rather than recomputed every frame.
        self._cloth_reference_mask: np.ndarray | None = None
        cloth_ref_path_str = config.detection.hough.cloth_reference_path
        if cloth_ref_path_str:
            cloth_ref_path = Path(cloth_ref_path_str)
            if cloth_ref_path.exists():
                loaded_cloth_ref = cv2.imread(str(cloth_ref_path), cv2.IMREAD_UNCHANGED)
                if loaded_cloth_ref is not None:
                    self._cloth_reference_mask = loaded_cloth_ref
                else:
                    logger.warning("could not load cloth reference: {}", cloth_ref_path)
        # Belt inspection window — driven by belt_inspect_signal (see _on_frame).
        # _manual_einlaufband_toggle: the simulated source; replaced by a PLC read later.
        # _belt_inspect_window_open: True while the inspection window is active.
        # _belt_window_fault_latched: True once a fault fires this window run; prevents
        #   a second verdict in the same Einlaufband run even if the belt clears briefly.
        self._manual_einlaufband_toggle: bool = False
        self._belt_inspect_window_open: bool = False
        self._belt_window_fault_latched: bool = False

        # OPC UA server (legacy server-mode bridge, dormant in production client mode).
        self._opcua = None
        # Production PLC client (OPC UA CLIENT to PLC server). None in demo mode.
        self._plc = None
        self._web = None

        # Views.
        self._sidebar = Sidebar()
        self._inference_view = InferenceView()
        self._capture_view = CaptureView()
        self._review_view = ReviewView()
        self._service_view = ServiceView(config)
        self._wizard_view = WizardView()

        self._stack = QStackedWidget()
        for v in (
            self._inference_view, self._capture_view, self._review_view,
            self._service_view, self._wizard_view,
        ):
            self._stack.addWidget(v)
        self._stack.setCurrentWidget(self._inference_view)

        central = QWidget()
        central.setObjectName("centralWidget")
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._sidebar)

        # Right-side wrapper: banner slot at top, views below.
        _right = QWidget()
        self._content_layout = QVBoxLayout(_right)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content_layout.addWidget(self._stack, 1)
        outer.addWidget(_right, 1)

        self.setCentralWidget(central)

        self._status = StatusBar(self)
        self.setStatusBar(self._status)
        self._refresh_status_bar()

        self._sidebar.set_active("inference")
        self._sidebar.nav_changed.connect(self._on_nav_changed)
        self._sidebar.mode_clicked.connect(self._on_mode_pill_clicked)
        self._wire_views()

        # Initial UI state.
        self._inference_view.set_profiles(
            self._cfg.profile_store().names(), self._cfg.app.active_profile,
        )
        self._inference_view.set_mode(Mode(self._cfg.app.mode))
        self._refresh_state(State.WAITING)
        self._sm.on_change(self._on_sm_change)

        # Startup banner — timer is wired inside _build_startup_banner().
        self._banner = self._build_startup_banner()

        # Start camera immediately so every view (Capture, Wizard, InferenceView)
        # shows live feed without requiring the user to press START first.
        self._start_camera()

        # Heartbeat: pumps the FAULT self-clear loop when the camera is off.
        # The detection pipeline drives FAULT self-clear when running; if the
        # operator stops the camera we still want SERVICE to be reachable, so
        # we leave this simple — no heartbeat needed.

    # ---------- public lifecycle ----------

    def attach_opcua(self, opcua) -> None:  # noqa: ANN001
        self._opcua = opcua
        if opcua is not None:
            opcua.on_tuchabzug_change = self._on_opcua_tuchabzug_change
        self._refresh_status_bar()

    def attach_plc_client(self, plc) -> None:  # noqa: ANN001
        """Attach the production PLC client. Called by main.py in production mode."""
        self._plc = plc
        if plc is not None:
            plc.on_tuchabzug_change = self._on_plc_tuchabzug_change
            plc.on_error_quit_change = self._on_plc_error_quit
        self._refresh_status_bar()

    def attach_web_server(self, web) -> None:  # noqa: ANN001
        self._web = web
        self._refresh_status_bar()

    def hide_startup_banner(self) -> None:
        self._banner.hide()
        self._banner.setFixedHeight(0)

    def shutdown(self) -> None:
        """Stop background threads. Called from main.py on close."""
        try:
            self._camera.stop()
        except Exception:  # noqa: BLE001
            logger.exception("camera.stop failed")
        try:
            self._inference_view.stop_recording_on_shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("inference_view recording stop failed")
        if self._recorder.is_recording:
            try:
                self._recorder.stop()
            except Exception:  # noqa: BLE001
                logger.exception("recorder.stop failed")
        if self._plc is not None:
            try:
                self._plc.stop()
            except Exception:  # noqa: BLE001
                logger.exception("plc.stop failed")
        if self._opcua is not None:
            try:
                self._opcua.stop()
            except Exception:  # noqa: BLE001
                logger.exception("opcua.stop failed")
        if self._web is not None:
            try:
                self._web.stop()
            except Exception:  # noqa: BLE001
                logger.exception("web.stop failed")

    def closeEvent(self, event) -> None:  # noqa: D401
        self.shutdown()
        super().closeEvent(event)

    # ---------- wiring ----------

    def _wire_views(self) -> None:
        iv = self._inference_view
        iv.start_clicked.connect(self._enable_detection)
        iv.stop_clicked.connect(self._disable_detection)
        iv.profile_changed.connect(self._on_profile_changed)
        iv.simulate_pulse_clicked.connect(self._on_simulate_pulse)
        iv.force_toggle_tuchabzug_clicked.connect(self._on_force_toggle_tuchabzug)
        iv.einlaufband_toggled.connect(self._on_toggle_einlaufband)

        cv = self._capture_view
        cv.snapshot_requested.connect(self._on_save_snapshot)
        cv.record_toggled.connect(self._on_record_toggled)

        rv = self._review_view
        rv.set_dirs(
            Path(self._cfg.storage.defect_image_dir), Path(self._cfg.storage.log_dir),
        )
        rv.export_logs_clicked.connect(self._on_export_logs)

        sv = self._service_view
        sv.mode_changed.connect(self._on_mode_changed)
        sv.language_changed.connect(self._on_language_changed)
        sv.orientation_changed.connect(self._on_orientation_changed)
        sv.profile_active_changed.connect(self._on_profile_changed)
        sv.profile_renamed.connect(self._on_profile_renamed)
        sv.profile_deleted.connect(self._on_profile_deleted)
        sv.web_settings_changed.connect(self._on_web_settings_changed)
        sv.pin_changed.connect(self._save_cfg)
        sv.learn_reference_requested.connect(self._on_learn_reference)
        sv.open_wizard_requested.connect(self._open_wizard)
        sv.entered_service.connect(self._sm.enter_service)
        sv.exited_service.connect(self._sm.exit_service)

        wz = self._wizard_view
        wz.cancelled.connect(self._on_wizard_cancelled)
        wz.completed.connect(self._on_wizard_completed)
        wz.apply_orientation_requested.connect(self._on_wizard_apply_orientation)
        wz.learn_reference_requested.connect(self._on_learn_reference)
        wz.roi_split_x_changed.connect(self._on_wizard_roi_changed)
        wz.transfer_line_x_changed.connect(self._on_wizard_tl_changed)
        wz.exposure_changed.connect(self._on_wizard_exposure_changed)
        wz.gain_changed.connect(self._on_wizard_gain_changed)
        wz.noise_threshold_changed.connect(self._on_wizard_noise_threshold_changed)
        wz.dough_is_darker_changed.connect(self._on_dough_is_darker_changed)
        wz.belt_crop_changed.connect(self._on_wizard_belt_crop_changed)
        wz.cloth_crop_changed.connect(self._on_wizard_cloth_crop_changed)
        wz.belt_dough_is_darker_changed.connect(self._on_belt_dough_is_darker_changed)
        # FIX 2: same handler as the Service page combo — single source of
        # truth, both controls stay in sync (see _on_detection_method_changed).
        wz.detection_method_changed.connect(self._on_detection_method_changed)
        wz.row_grouping_changed.connect(self._on_row_grouping_changed)
        wz.save_cloth_reference_requested.connect(self._on_save_cloth_reference)
        wz.transfer_bridge_width_changed.connect(self._on_wizard_bridge_width_changed)
        wz.grid_columns_changed.connect(self._on_wizard_grid_columns_changed)
        wz.grid_rows_changed.connect(self._on_wizard_grid_rows_changed)

        sv.dough_is_darker_changed.connect(self._on_dough_is_darker_changed)
        sv.belt_dough_is_darker_changed.connect(self._on_belt_dough_is_darker_changed)
        sv.show_belt_mask_changed.connect(lambda v: setattr(self, "_belt_mask_enabled", v))
        sv.show_cloth_mask_changed.connect(lambda v: setattr(self, "_cloth_mask_enabled", v))
        sv.open_crop_wizard_requested.connect(lambda: self._open_wizard(start_step=3))

        sv.detection_method_changed.connect(self._on_detection_method_changed)
        sv.fill_mask_holes_changed.connect(self._on_fill_mask_holes_changed)
        sv.capture_bg_reference_requested.connect(self._on_capture_bg_reference)
        sv.open_cloth_reference_requested.connect(lambda: self._open_wizard(start_step=6))

        sv.frame_source_changed.connect(self._on_frame_source_changed)
        sv.playback_folder_changed.connect(self._on_playback_folder_changed)
        sv.playback_loop_changed.connect(self._on_playback_loop_changed)
        sv.playback_play_requested.connect(self._on_playback_play)
        sv.playback_pause_requested.connect(self._on_playback_pause)
        sv.playback_step_requested.connect(self._on_playback_step)

    # ---------- nav ----------

    def _show_view(self, nav_id: str) -> None:
        targets = {
            "inference": self._inference_view,
            "capture": self._capture_view,
            "review": self._review_view,
            "service": self._service_view,
            "wizard": self._wizard_view,
        }
        view = targets.get(nav_id)
        if view is None:
            return
        if nav_id == "review":
            self._review_view.refresh()
        self._stack.setCurrentWidget(view)
        self._sidebar.set_active(nav_id if nav_id != "wizard" else "service")

    def _on_nav_changed(self, nav_id: str) -> None:
        if nav_id != "service":
            self._service_view.lock()
        self._show_view(nav_id)

    def _on_mode_pill_clicked(self) -> None:
        # Quick toggle; SERVICE view is the canonical place but a click on the
        # mode pill is also wired for convenience.
        new = "production" if self._cfg.app.mode == "demo" else "demo"
        self._on_mode_changed(new)

    # ---------- camera frame handling ----------

    def _start_camera(self) -> None:
        if self._camera_running:
            return
        self._camera.apply_profile(self._active_profile_cache)
        self._camera.start(self._frame_bridge.emit_frame)
        self._camera_running = True
        self._refresh_status_bar()

    def _stop_camera(self) -> None:
        if not self._camera_running:
            return
        self._camera.stop()
        self._camera_running = False
        self._refresh_status_bar()

    def _enable_detection(self) -> None:
        """START button: enable blob-detection pipeline; camera already runs."""
        self._detection_enabled = True

    def _disable_detection(self) -> None:
        """STOP button: disable blob-detection pipeline; camera keeps running."""
        self._detection_enabled = False
        self._sm.force_reset_to_waiting()
        self._belt_fault_debounce_count = 0
        self._belt_fault_armed = True
        self._belt_clear_count = 0
        self._prev_tuchabzug_running = False
        self._row_phase = RowPhase.IDLE
        self._line_clear_debounce_count = 0
        self._transfer_timeout_warned = False
        self._row_line_tracker.reset()
        self._row_group_stop_active = False
        self._row_lines = []
        self._row_count = 0
        self._current_row_centroids = []
        self._detection_band = None
        self._last_hough_time = 0.0
        self._cached_cloth_detections = []
        self._cached_cloth_highlight = []
        self._cached_tripwire_occupied = False
        self._active_bridge_width_px = 0
        self._detection_zone_outer_x = None
        self._active_row_index = 0
        self._row_status = ["WAITING"] * max(0, self._cfg.detection.grid_rows)
        self._committed_boundary_x = None
        self._cycle_partial_warned = False
        self._cycle_front_row = []
        self._cycle_row2 = []
        self._cycle_ref_tangent_x = None
        self._cycle_log_next_time = 0.0
        self._last_known_row_leading_x = None
        self._memory_frame_count = 0
        self._belt_inspect_window_open = False
        self._belt_window_fault_latched = False
        self._clear_sticky_overlay()
        self._push_signals_to_opcua()

    def _refresh_profile_cache(self) -> None:
        """Rebuild the active-profile cache after any config change that affects it."""
        self._active_profile_cache = self._cfg.active_profile()
        if self._last_frame_w > 0:
            p = self._active_profile_cache
            self._inference_view.set_roi_info(p.roi_split_x, self._last_frame_w, p.transfer_line_x)

    @Slot(object)
    def _on_frame(self, frame: np.ndarray) -> None:
        self._frame_bridge._processing = True
        _now_mono = time.monotonic()
        if _now_mono - self._dropped_frames_last_log >= 5.0:
            _dropped = self._frame_bridge._dropped
            if _dropped > 0:
                logger.info(
                    "Dropped {} frames in last 5s (processing can't keep up)",
                    _dropped,
                )
            self._frame_bridge._dropped = 0
            self._dropped_frames_last_log = _now_mono
        _t0 = time.perf_counter()
        self._frame_count += 1

        # CALIBRATION TOOLING: hand the raw frame (pre-pipeline, pre-split,
        # pre-detection) to the recorder. submit() is non-blocking — it queues
        # a copy onto a background writer thread and drops it under
        # backpressure, so this never stalls _on_frame. No effect when not
        # recording (is_recording check is the first thing submit() does).
        if self._recorder.is_recording:
            self._recorder.submit(frame)

        self._latest_frame = frame
        self._capture_view.set_frame(frame)

        current_view = self._stack.currentWidget()
        if current_view is self._service_view:
            self._service_view.set_preview_frame(frame)
        elif current_view is self._wizard_view:
            self._wizard_view.set_preview_frame(frame)

        profile = self._active_profile_cache

        if frame.shape[1] != self._last_frame_w:
            self._last_frame_w = frame.shape[1]
            self._inference_view.set_roi_info(
                profile.roi_split_x, self._last_frame_w, profile.transfer_line_x
            )

        _t_split = time.perf_counter()
        try:
            belt, cloth = self._pipeline.split_rois(frame, profile)
        except ValueError:
            self._frame_bridge._processing = False
            return
        _split_ms = (time.perf_counter() - _t_split) * 1000.0

        transfer_local: int = profile.transfer_line_x - profile.roi_split_x
        highlight: list[tuple[float, float]] = []
        cloth_detections: list = []
        belt_detections: list = []
        _tripwire_occupied = False
        _diag_row_profile = None  # DIAGNOSTIC: see InspectionPipeline.DIAGNOSTIC_ROW_PROFILE
        _diag_row_profile_scale = 1.0

        belt_crop_rect = self._pipeline.crop_rect_for(belt, profile.belt_crop)
        cloth_crop_rect = self._pipeline.crop_rect_for(cloth, profile.cloth_crop)

        _t_tripwire_ms = 0.0
        _t_detect_ms = 0.0     # SECTION 1: cloth detection (+ mask/gating) wall time
        _t_rowgroup_ms = 0.0   # SECTION 1: row-grouping / tripwire-edge decision time
        _t_belt_ms = 0.0
        _detection_ran = False  # SECTION 1: did a detection pass run this frame?

        if self._detection_enabled and self._sm.state != State.SERVICE:
            snap = self._sm.snapshot()

            # FIX 1: pre-compute whether belt inspection is due this frame so we
            # can stagger: avoid running cloth Hough (~25ms) AND belt (~15ms) on
            # the same frame when possible.
            _needs_belt_precheck = self._belt_inspect_window_open or snap.fault_active
            _belt_due_this_frame = (
                _needs_belt_precheck
                and self._frame_count % self._cfg.inspection.belt_check_interval == 0
            )

            # ---- Check 1: Cloth tripwire — gated on TuchabzugRunning (UNCHANGED) ----
            # _prev_tuchabzug_running is updated unconditionally (outside the
            # running-gated block) so the False→True rising edge is detected on
            # EVERY cycle, not just the first.  Previously this lived inside the
            # if-block and was never reset when cloth stopped, causing cycle-2+
            # tangent-stop resets to be skipped.
            _is_rising_edge = snap.tuchabzug_running and not self._prev_tuchabzug_running
            self._prev_tuchabzug_running = snap.tuchabzug_running
            if snap.tuchabzug_running:
                # Reset per-cycle state on the very first frame of a new cloth pull.
                if _is_rising_edge:
                    self._reset_tracking_session()
                    if self._cfg.detection.use_row_grouping:
                        _grid_rows = self._cfg.detection.grid_rows
                        logger.info(
                            "Row-SM rising edge: active_row={} status_before={}",
                            self._active_row_index,
                            self._row_status,
                        )
                        if (
                            self._active_row_index < _grid_rows
                            and self._row_status[self._active_row_index] == "STOPPED"
                        ):
                            _old_idx = self._active_row_index
                            self._row_status[self._active_row_index] = "DONE"
                            self._active_row_index += 1
                            logger.info(
                                "Row-SM ADVANCE: row {} marked DONE -> "
                                "new active_row={} row_status={}",
                                _old_idx + 1,
                                self._active_row_index,
                                self._row_status,
                            )
                            self._row_advance_pending_log = True
                        if self._active_row_index >= _grid_rows:
                            self._active_row_index = 0
                            self._row_status = ["WAITING"] * max(0, _grid_rows)
                            logger.info(
                                "Row-SM FULL RESET: all rows complete -> "
                                "active_row_index=0, row_status reset to {}",
                                self._row_status,
                            )
                            _old_bnd = self._committed_boundary_x
                            self._committed_boundary_x = None
                            if _old_bnd is not None:
                                logger.info(
                                    "Committed boundary RESET: cleared (was {}) for new cycle",
                                    int(round(_old_bnd)),
                                )

                # FIX 1: rate-limit Hough to hough_interval_ms between runs.
                # Stagger: if belt is also due this frame, defer Hough unless it
                # is overdue (hasn't run in >2× the interval — run it anyway then).
                _hough_interval_s = self._cfg.detection.hough.hough_interval_ms / 1000.0
                _hough_elapsed = _now_mono - self._last_hough_time
                _hough_due = (
                    _hough_elapsed >= _hough_interval_s
                    and self._frame_count % self._cfg.inspection.tripwire_check_interval == 0
                )
                if _hough_due and _belt_due_this_frame and _hough_elapsed < 2.0 * _hough_interval_s:
                    _hough_due = False  # defer by one frame (stagger)

                _t_tw = time.perf_counter()
                if _hough_due:
                    _detection_ran = True
                    self._last_hough_time = _now_mono
                    _t_d0 = time.perf_counter()
                    r = self._pipeline.run_cloth_tracking(
                        frame, profile, self._cfg.detection,
                        bg_reference_cloth=self._bg_reference_cloth,
                    )
                    _t_detect_ms = (time.perf_counter() - _t_d0) * 1000.0
                    # Cache for reuse between Hough runs.
                    self._cached_cloth_detections = r.detections
                    self._cached_cloth_highlight = list(r.front_row_centroids)
                    self._cached_tripwire_occupied = r.tripwire_occupied
                    cloth_detections = r.detections
                    highlight = r.front_row_centroids
                    _tripwire_occupied = r.tripwire_occupied
                    if r.inference_ms > 50.0:
                        logger.debug("cloth_tracking slow: {:.0f} ms", r.inference_ms)
                    # Firing path: row grouping (per-row stop) when enabled,
                    # otherwise the unchanged single-line tripwire.
                    _t_r0 = time.perf_counter()
                    if self._cfg.detection.use_row_grouping:
                        self._apply_tangent_stop_edge(r, profile)
                    else:
                        self._apply_tripwire_edge(r, profile)
                    _t_rowgroup_ms = (time.perf_counter() - _t_r0) * 1000.0

                    # --- DIAGNOSTIC: row-profile bump/valley log (two-rows-at-once
                    # investigation). Read-only: does not feed any decision.
                    # Throttled to ~3 Hz independent of tripwire_check_interval.
                    _diag_row_profile = r.row_profile
                    _diag_row_profile_scale = r.row_profile_scale
                    if r.row_profile is not None:
                        _now_diag = time.monotonic()
                        if _now_diag - self._diag_row_profile_last_log >= 0.33:
                            self._diag_row_profile_last_log = _now_diag
                            _bumps, _valley_pct = InspectionPipeline.analyze_row_profile(
                                r.row_profile
                            )
                            logger.info(
                                "DIAGNOSTIC row profile: bumps={} deepest_valley={:.0f}% of peak",
                                _bumps, _valley_pct,
                            )
                    # --- END DIAGNOSTIC ---
                else:
                    # Hough not due this frame — reuse cached cloth results for display.
                    cloth_detections = self._cached_cloth_detections
                    highlight = self._cached_cloth_highlight
                    _tripwire_occupied = self._cached_tripwire_occupied
                _t_tripwire_ms = (time.perf_counter() - _t_tw) * 1000.0
                # Warn once if the row doesn't clear within transfer_timeout_ms.
                if self._row_phase == RowPhase.TRANSFERRING and not self._transfer_timeout_warned:
                    _elapsed_ms = (time.monotonic() - self._transfer_start_time) * 1000.0
                    if _elapsed_ms > self._cfg.inspection.transfer_timeout_ms:
                        logger.warning(
                            "transfer timeout: row not cleared after {:.0f}ms", _elapsed_ms,
                        )
                        self._transfer_timeout_warned = True

            # ---- Check 2: Belt inspection — driven by belt_inspect_signal ----
            # Belt and cloth run on INDEPENDENT motors; the belt window is NOT
            # tied to TuchabzugRunning.  It is controlled by a separate signal.
            #
            # Primary source is the PLC's toext_Einlaufband_running; the manual
            # toggle remains as a fallback/override when no PLC is attached
            # (demo mode, or production before the PLC connects).
            belt_inspect_signal = self._read_belt_inspect_signal()

            if belt_inspect_signal and not self._belt_inspect_window_open:
                self._open_belt_inspect_window()
            elif not belt_inspect_signal and self._belt_inspect_window_open:
                self._close_belt_inspect_window()

            # Run belt detection while the window is open OR while clearing a fault.
            # Skipping in all other states saves 15-25 ms/frame (unchanged from before).
            _needs_belt = self._belt_inspect_window_open or snap.fault_active
            _t_belt = time.perf_counter()
            if _needs_belt and self._frame_count % self._cfg.inspection.belt_check_interval == 0:
                result = self._pipeline.run_belt_inspection(
                    frame, profile, unknown_is_fault=self._cfg.inspection.unknown_is_fault,
                )
                belt_detections = result.detections
                logger.debug(
                    "Belt: {} blob(s), verdict={}",
                    len(result.detections), result.verdict.value,
                )
                if result.inference_ms > 50.0:
                    logger.debug("belt_inspection slow: {:.0f} ms", result.inference_ms)

                # Fault RAISING: only inside the open window and only once per run.
                # _belt_window_fault_latched prevents a second verdict even if the
                # belt clears briefly and goes dirty again within the same run.
                if self._belt_inspect_window_open and not self._belt_window_fault_latched:
                    _was_armed = self._belt_fault_armed
                    self._apply_belt_fault_debounce(result, belt_roi=belt)
                    if _was_armed and not self._belt_fault_armed:
                        # Fault just fired (armed→disarmed) — latch for this window run.
                        self._belt_window_fault_latched = True

                # Fault CLEARING: always while fault_active, independent of window.
                # FIX 4 (single shared latched fault state, UI <-> PLC): in
                # production, the clean-frame self-clear below is DISABLED —
                # otherwise FaultActive could clear here while the PLC's
                # ext_error stays latched at 2 until the operator acks, which
                # is exactly the drift this fix removes. Production's ONLY
                # clear path is the operator ack (ext_error_quit rising edge,
                # see _on_plc_error_quit -> StateMachine.acknowledge_fault).
                # Demo has no PLC ack signal, so it keeps the original
                # self-clear-after-N-clean-frames behavior.
                snap = self._sm.snapshot()
                _production_latched = self._cfg.app.mode == "production" and self._plc is not None
                if snap.fault_active:
                    if not result.verdict.is_fault:
                        if not _production_latched and self._sm.handle_clean_frame_in_fault():
                            self._on_fault_cleared()
                    else:
                        self._sm.handle_dirty_frame_in_fault()

                # Sticky overlay for display after the inspection window closes.
                self._sticky_belt_detections = list(result.detections)
                self._sticky_belt_crop_rect = belt_crop_rect
                hold_ms = max(0, int(
                    self._cfg.inspection.inspection_overlay_hold_seconds * 1000
                ))
                if hold_ms > 0:
                    self._sticky_overlay_timer.start(hold_ms)
            _t_belt_ms = (time.perf_counter() - _t_belt) * 1000.0

        _t_tripwire_ms = _t_detect_ms + _t_rowgroup_ms
        if _t_tripwire_ms > 0.0 or _t_belt_ms > 0.0:
            self._latest_pipeline_ms = _t_tripwire_ms + _t_belt_ms

        # snap_display is always needed for the perf log; capture it before the
        # display gate so it is defined on both the refresh and skip paths.
        snap_display = self._sm.snapshot()

        # Display: throttled to _DISPLAY_INTERVAL_S.
        # Detection runs at full camera rate above; only the QImage conversion
        # and QPixmap scaling (the expensive part) is held back.
        _t_display = time.perf_counter()
        if _now_mono - self._last_display_refresh >= _DISPLAY_INTERVAL_S:
            self._last_display_refresh = _now_mono

            # Show sticky overlay when cloth is stopped and timer is active.
            _disp_belt_dets = belt_detections
            _disp_belt_crop = belt_crop_rect
            if (
                not _disp_belt_dets
                and self._sticky_overlay_timer.isActive()
                and not snap_display.tuchabzug_running
            ):
                _disp_belt_dets = self._sticky_belt_detections
                _disp_belt_crop = self._sticky_belt_crop_rect or belt_crop_rect

            _belt_debug_mask = None
            if self._belt_mask_enabled:
                try:
                    belt_diag_profile = profile.model_copy(
                        update={"dough_is_darker": profile.belt_dough_is_darker}
                    )
                    _cropped_belt, _ = self._pipeline.apply_crop(belt, profile.belt_crop)
                    _belt_debug_mask, _ = self._detector.compute_binary_mask(
                        _cropped_belt, belt_diag_profile
                    )
                except Exception:  # noqa: BLE001
                    _belt_debug_mask = None

            _cloth_debug_mask = None
            if self._cloth_mask_enabled:
                try:
                    _cropped_cloth, _ = self._pipeline.apply_crop(cloth, profile.cloth_crop)
                    method = self._cfg.detection.method
                    if method == "bg_subtract":
                        _cropped_ref = None
                        if (
                            self._bg_reference_cloth is not None
                            and self._bg_reference_cloth.shape[:2] == cloth.shape[:2]
                        ):
                            _cropped_ref, _ = self._pipeline.apply_crop(
                                self._bg_reference_cloth, profile.cloth_crop
                            )
                        _cloth_debug_mask = self._detector.compute_bg_subtract_mask(
                            _cropped_cloth, _cropped_ref,
                            threshold=self._cfg.detection.bg_subtract.threshold,
                        )
                    elif method == "hough":
                        # FIX 1 diagnostic: shows exactly the bright-cloth
                        # region Hough is allowed to see (see detect_hough).
                        _h_cfg = self._cfg.detection.hough
                        _cloth_debug_mask = self._detector.compute_hough_cloth_mask(
                            _cropped_cloth, profile,
                            brightness_threshold=_h_cfg.cloth_brightness_threshold,
                            downscale_factor=_h_cfg.downscale_factor,
                        )
                    else:
                        # Fill-holes preview only makes sense for blob (the
                        # other methods ignore it — see _DetectionSection).
                        _fill = method == "blob" and self._cfg.detection.fill_mask_holes
                        _cloth_debug_mask, _ = self._detector.compute_binary_mask(
                            _cropped_cloth, profile,
                            fill_holes=_fill,
                            fill_holes_kernel=self._cfg.detection.fill_mask_holes_kernel,
                        )
                except Exception:  # noqa: BLE001
                    _cloth_debug_mask = None

            # Detection zone outer boundary for display (approach-side edge in
            # cloth-local coords). Computed from the size-adaptive bridge half
            # plus the configured zone extension. Drawn as an orange dashed line.
            if self._cfg.detection.detection_zone_width_px > 0:
                _zone_bridge_half = (
                    self._active_bridge_width_px / 2.0
                    if self._active_bridge_width_px > 0
                    else profile.transfer_bridge_width_px / 2.0
                )
                self._detection_zone_outer_x = int(
                    transfer_local + _zone_bridge_half
                    + self._cfg.detection.detection_zone_width_px
                )
            else:
                self._detection_zone_outer_x = None

            # Show cached cloth detections even when the cloth is stopped so
            # tangent lines remain visible on a static scene for pre-transfer
            # verification.  Does not affect stop decisions (those only run
            # when tuchabzug_running is True).
            if not cloth_detections and self._cached_cloth_detections:
                cloth_detections = self._cached_cloth_detections
                if not highlight:
                    highlight = self._cached_cloth_highlight

            self._inference_view.set_belt_frame(
                belt, _disp_belt_dets, px_per_mm=1.0, crop_rect=_disp_belt_crop,
                debug_mask=_belt_debug_mask,
            )
            # Grid visualization: compute row assignments from current
            # cloth_detections for coloring in _draw_detection.
            _row_count_display: int | None = None
            if self._cfg.detection.use_row_grouping and cloth_detections:
                # Count-based grouping with boundary filter (same filter as stop decision).
                # Excluded pieces (X <= committed_boundary_x) are drawn gray (-1);
                # only eligible pieces participate in the sort+slice row coloring.
                _disp_bnd = self._committed_boundary_x
                _disp_eligible = [
                    d for d in cloth_detections
                    if _disp_bnd is None or leading_edge_x(d) > _disp_bnd
                ]
                _disp_excluded = [
                    d for d in cloth_detections
                    if _disp_bnd is not None and leading_edge_x(d) <= _disp_bnd
                ]
                _grouped = slice_rows_by_count(_disp_eligible, self._cfg.detection.grid_columns)
                _row_count_display = len(_grouped)
                # Map detection object → index for O(1) lookup; assign row colours.
                _det_to_idx = {id(det): i for i, det in enumerate(cloth_detections)}
                _grid_row_asgn: dict[int, int] = {}
                for _det in _disp_excluded:
                    _di = _det_to_idx.get(id(_det))
                    if _di is not None:
                        _grid_row_asgn[_di] = -1  # gray — boundary-excluded
                for _gi, _grp in enumerate(_grouped[:2]):
                    for _det in _grp:
                        _di = _det_to_idx.get(id(_det))
                        if _di is not None:
                            _grid_row_asgn[_di] = _gi + 1  # 1 = front/magenta, 2 = second/cyan
                _ref_x = self._cycle_ref_tangent_x
                _ref_lbl = f"{int(_ref_x)}" if _ref_x is not None else "—"
                _grs = self._cfg.detection.grid_rows
                _grid_label = (
                    f"Active Row: {self._active_row_index + 1}/"
                    f"{_grs if _grs > 0 else '?'} "
                    f"| Status: {self._row_status} | Ref: {_ref_lbl}px"
                )
            else:
                _grid_row_asgn = None
                _ref_x = None
                _grid_label = None

            self._inference_view.set_cloth_frame(
                cloth,
                detections=cloth_detections,
                transfer_line_local_x=transfer_local if 0 <= transfer_local < cloth.shape[1] else None,
                highlight_centroids=highlight,
                px_per_mm=1.0,
                crop_rect=cloth_crop_rect,
                tripwire_half_width_px=profile.tripwire_half_width_px,
                tripwire_occupied=_tripwire_occupied,
                # FIX 2: use the size-adaptive bridge width for drawing; fall
                # back to the configured value before the first detection runs.
                transfer_bridge_width_px=(
                    self._active_bridge_width_px
                    if self._active_bridge_width_px > 0
                    else profile.transfer_bridge_width_px
                ),
                # DIAGNOSTIC ONLY — see InspectionPipeline.DIAGNOSTIC_ROW_PROFILE.
                row_profile=_diag_row_profile,
                row_profile_x_offset=cloth_crop_rect[0],
                row_profile_scale=_diag_row_profile_scale,
                # Tangent-stop path draws per-piece leading-edge lines inside
                # _draw_detection; vertical row-position lines not shown.
                row_lines=None,
                row_count=_row_count_display,
                # Bridge IS the detection band (FIX 2 unified): don't double-draw
                # the same region. Pass None so only the bridge overlay renders.
                detection_band=None,
                current_row_centroids=None,
                detection_zone_outer_x=self._detection_zone_outer_x,
                grid_row_assignments=_grid_row_asgn,
                grid_ref_tangent_x=_ref_x,
                grid_label=_grid_label,
                debug_mask=_cloth_debug_mask,
            )
            self._inference_view.set_inference_ms(self._latest_pipeline_ms)
            self._inference_view.set_plc_signals(
                tuchabzug_running=snap_display.tuchabzug_running,
                stop_tuchabzug=snap_display.stop_tuchabzug,
                fault_active=snap_display.fault_active,
                einlaufband_running=self._read_belt_inspect_signal(),
            )
            self._inference_view.set_belt_inspect_state(self._belt_inspect_window_open)
            self._inference_view.set_cloth_row_phase(self._row_phase)
            self._update_banner_from_signals(snap_display)
        _display_ms = (time.perf_counter() - _t_display) * 1000.0

        if self._web is not None:
            now = time.monotonic()
            if now - self._last_web_encode >= 0.5:
                self._last_web_encode = now
                _tw = time.perf_counter()
                try:
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        self._web.set_latest_jpeg(bytes(buf))
                except Exception:  # noqa: BLE001
                    pass
                _web_ms = (time.perf_counter() - _tw) * 1000.0
                if _web_ms > 50.0:
                    logger.warning("web JPEG encode slow: {:.0f} ms", _web_ms)

        _frame_ms = (time.perf_counter() - _t0) * 1000.0

        # SECTION 1: per-stage timing, averaged over a 1-second window and
        # emitted ONCE PER SECOND at INFO (throttled — a per-frame INFO would
        # cause stderr backpressure; this is one line/sec). Makes the bottleneck
        # visible (detection vs mask/gating-inclusive vs row-grouping vs UI
        # render) and reports the real detection frequency + frame-drop rate.
        self._perf_totals["split"] += _split_ms
        self._perf_totals["detect"] += _t_detect_ms
        self._perf_totals["rowgroup"] += _t_rowgroup_ms
        self._perf_totals["belt"] += _t_belt_ms
        self._perf_totals["display"] += _display_ms
        self._perf_totals["total"] += _frame_ms
        self._perf_frames += 1
        if _detection_ran:
            self._perf_detect_count += 1
        if _now_mono - self._perf_last_log >= _PERF_LOG_INTERVAL_S and self._perf_frames > 0:
            n = self._perf_frames
            window_s = max(1e-3, _now_mono - self._perf_last_log)
            detect_hz = self._perf_detect_count / window_s
            if self._detection_enabled:
                logger.info(
                    "perf 1s ({} frames, {:.1f} fps): total={:.1f}ms "
                    "detect={:.1f}ms rowgroup={:.1f}ms belt={:.1f}ms "
                    "display={:.1f}ms split={:.1f}ms | detection {:.1f}/s",
                    n, n / window_s,
                    self._perf_totals["total"] / n,
                    self._perf_totals["detect"] / n,
                    self._perf_totals["rowgroup"] / n,
                    self._perf_totals["belt"] / n,
                    self._perf_totals["display"] / n,
                    self._perf_totals["split"] / n,
                    detect_hz,
                )
            for k in self._perf_totals:
                self._perf_totals[k] = 0.0
            self._perf_frames = 0
            self._perf_detect_count = 0
            self._perf_last_log = _now_mono

        logger.debug(
            "_on_frame: total={:.1f}ms tripwire={:.1f}ms belt={:.1f}ms "
            "running={} fault={}",
            _frame_ms, _t_tripwire_ms, _t_belt_ms,
            snap_display.tuchabzug_running, snap_display.fault_active,
        )
        if _frame_ms > 50.0:
            logger.debug(
                "_on_frame overran: {:.0f} ms (detection={}, running={}, fault={})",
                _frame_ms, self._detection_enabled,
                snap_display.tuchabzug_running, snap_display.fault_active,
            )
        self._frame_bridge._processing = False

    def _open_belt_inspect_window(self) -> None:
        self._belt_inspect_window_open = True
        self._belt_window_fault_latched = False
        self._belt_fault_armed = True
        self._belt_fault_debounce_count = 0
        self._belt_clear_count = 0
        logger.info("Belt inspection window OPEN")

    def _close_belt_inspect_window(self) -> None:
        self._belt_inspect_window_open = False
        logger.info(
            "Belt inspection window CLOSED — fault_latched={}",
            self._belt_window_fault_latched,
        )

    def _on_toggle_einlaufband(self, active: bool) -> None:
        self._manual_einlaufband_toggle = active

    def _read_belt_inspect_signal(self) -> bool:
        """Resolve the belt-inspection-window source.

        Primary source: the PLC's toext_Einlaufband_running. Falls back to the
        UI's manual toggle in demo mode or whenever no PLC client is attached.
        """
        if self._plc is not None and self._cfg.app.mode == "production":
            try:
                return self._plc.read_einlaufband_running()
            except Exception:  # noqa: BLE001
                logger.exception("plc.read_einlaufband_running failed")
        return self._manual_einlaufband_toggle

    def _apply_tripwire_edge(self, r: object, profile: "ProductProfile") -> None:
        """Drive the row-at-line state machine from the per-frame tripwire reading.

        StopTuchabzug fires ONLY on IDLE → AT_LINE (a genuinely new row).
        During TRANSFERRING the tripwire is suppressed — the same row cannot
        trigger twice.  TRANSFERRING → IDLE requires line_clear_debounce_frames
        consecutive clear frames so a partially transferred row doesn't re-arm
        prematurely.
        """
        occupied_now: bool = r.tripwire_occupied  # type: ignore[attr-defined]
        debounce = profile.tripwire_debounce_frames

        # Standard per-edge debounce — reset clear counter on any occupancy flip.
        if occupied_now == self._tripwire_candidate:
            self._tripwire_debounce_count += 1
        else:
            self._tripwire_candidate = occupied_now
            self._tripwire_debounce_count = 1
            self._line_clear_debounce_count = 0

        if self._tripwire_debounce_count < debounce:
            return

        stable: bool = self._tripwire_candidate  # debounced occupancy

        if stable:
            # ---- line occupied ----
            if self._row_phase == RowPhase.IDLE:
                # New row reached the line: IDLE → AT_LINE, fire StopTuchabzug.
                self._row_phase = RowPhase.AT_LINE
                self._tripwire_prev_stable = True
                logger.info(
                    "Row AT_LINE: StopTuchabzug TRUE (new row), "
                    "occupancy={:.1%} at transfer_line_local={}",
                    r.tripwire_occupancy,  # type: ignore[attr-defined]
                    profile.transfer_line_x - profile.roi_split_x,
                )
                self._sm.handle_stop_tuchabzug_trigger()
                self._push_signals_to_opcua()
                self._request_plc_stop_pulse()
            # AT_LINE: already handled — do nothing.
            # TRANSFERRING: row still on line during transfer — suppressed.

        else:
            # ---- line clear ----
            if self._row_phase == RowPhase.TRANSFERRING:
                # Count consecutive clear frames; transition to IDLE once stable.
                self._line_clear_debounce_count += 1
                if self._line_clear_debounce_count >= self._cfg.inspection.line_clear_debounce_frames:
                    self._row_phase = RowPhase.IDLE
                    self._tripwire_prev_stable = False
                    self._line_clear_debounce_count = 0
                    self._transfer_timeout_warned = False
                    logger.info("Line cleared: IDLE, tripwire re-armed")
                    self._push_signals_to_opcua()
            elif self._tripwire_prev_stable:
                # Falling edge outside TRANSFERRING (e.g., row retreated from AT_LINE).
                self._tripwire_prev_stable = False
                if self._row_phase == RowPhase.AT_LINE:
                    self._row_phase = RowPhase.IDLE
                logger.info("StopTuchabzug FALSE: line cleared, tripwire re-armed")
                self._sm.handle_stop_tuchabzug_clear()
                self._push_signals_to_opcua()

    def _band_filter(
        self, detections: list, transfer_x: float, effective_width: float
    ) -> list:
        """FIX 2 (UNIFIED): the detection band IS the transfer bridge — a
        symmetric region of width effective_width centered on transfer_x.
        band.enabled is still the master on/off; ahead_px/behind_px are no
        longer used. Records the band extent for the draw overlay (which
        coincides with the drawn bridge, so no double-draw is needed).
        """
        band = self._cfg.detection.band
        if not band.enabled:
            self._detection_band = None
            return detections
        half = effective_width / 2.0
        x_left = transfer_x - half
        x_right = transfer_x + half
        self._detection_band = (int(x_left), int(x_right))
        return [d for d in detections if x_left <= leading_edge_x(d) <= x_right]

    def _apply_tangent_stop_edge(self, r: object, profile: "ProductProfile") -> None:
        """USE_ROW_GROUPING=ON firing path — explicit per-row state machine.

        Each frame while TuchabzugRunning is True:
        1. Skip entirely if the cycle is complete (_active_row_index >= grid_rows).
        2. Look ONLY at grouped_rows[_active_row_index] for the stop decision.
        3. Fire StopTuchabzug when that row's leftmost tangent <= transfer_x
           and its status is WAITING; set status to STOPPED.
        Row advancement (STOPPED -> DONE, index++) happens in the rising-edge
        handler. Full-cycle reset (index -> 0, all WAITING) also happens there
        once all rows are DONE.
        """
        transfer_x = float(profile.transfer_line_x - profile.roi_split_x)
        all_detections = r.detections  # type: ignore[attr-defined]

        raw_diameter = median_piece_diameter(all_detections) or float(profile.expected_width_px)
        effective_width = max(float(profile.transfer_bridge_width_px), raw_diameter)
        self._active_bridge_width_px = int(effective_width)

        grid_cols = self._cfg.detection.grid_columns
        grid_rows = self._cfg.detection.grid_rows

        # Boundary filter: exclude pieces already handled by a previously-fired row.
        # Pieces with leading_edge_x <= committed_boundary_x are leftovers from a
        # row whose stop already fired and must not be regrouped as the new active row.
        _bnd_x = self._committed_boundary_x
        _eligible = [d for d in all_detections if _bnd_x is None or leading_edge_x(d) > _bnd_x]
        _excluded = [d for d in all_detections if _bnd_x is not None and leading_edge_x(d) <= _bnd_x]
        logger.info(
            "Boundary filter: {} pieces detected, {} excluded (X<=boundary {}), "
            "{} eligible for grouping",
            len(all_detections),
            len(_excluded),
            int(round(_bnd_x)) if _bnd_x is not None else "none",
            len(_eligible),
        )
        if _excluded:
            _all_sorted_for_cand = sorted(all_detections, key=leading_edge_x)
            _candidate_ids = {id(d) for d in _all_sorted_for_cand[:grid_cols]}
            for _excl_d in _excluded:
                if id(_excl_d) in _candidate_ids:
                    logger.info(
                        "Boundary filter EXCLUDED piece at X={} from grouping "
                        "(would have been row candidate)",
                        int(round(leading_edge_x(_excl_d))),
                    )

        grouped_rows = slice_rows_by_count(_eligible, grid_cols)

        # Sort-slice grouping log (unthrottled — diagnostic).
        _log_xs = [int(round(leading_edge_x(d))) for d in sorted(_eligible, key=leading_edge_x)]
        _log_groups = [[int(round(leading_edge_x(d))) for d in grp] for grp in grouped_rows]
        logger.info(
            "Sort-slice grouping: {} pieces detected, sorted X={}, sliced into groups_of_{}={}",
            len(_eligible), _log_xs, grid_cols, _log_groups,
        )

        active_idx = self._active_row_index

        # --- Step 1: cycle complete — skip stop evaluation ---
        if active_idx >= grid_rows:
            self._cycle_front_row = []
            self._cycle_row2 = []
            self._cycle_ref_tangent_x = None
            _now_log = time.monotonic()
            if _now_log >= self._cycle_log_next_time:
                self._cycle_log_next_time = _now_log + 0.2
                logger.info(
                    "Row-SM: cycle already complete, waiting for full reset "
                    "(active_row_index={} >= grid_rows={})",
                    active_idx, grid_rows,
                )
            return

        # --- Step 2: look ONLY at the active row's group ---
        # Always use grouped_rows[0]: once the previous row's pieces leave the
        # cloth the list shrinks, so the current active row is always the
        # front-most (smallest tangent) group regardless of active_row_index.
        active_row = grouped_rows[0] if grouped_rows else []
        row2 = grouped_rows[1] if len(grouped_rows) > 1 else []

        # Partial-detection warning: fires once when active row smaller than expected.
        grid_cols = self._cfg.detection.grid_columns
        if not self._cycle_partial_warned and 0 < len(active_row) < grid_cols:
            self._cycle_partial_warned = True
            logger.warning(
                "Partial detection: expected {} pieces in active row {}, got {} "
                "(groups={}) — proceeding",
                grid_cols, active_idx, len(active_row), len(grouped_rows),
            )

        # Leftmost tangent of the active row
        min_tang_x = min(leading_edge_x(d) for d in active_row) if active_row else None

        # ---- DIAGNOSTIC: group-selection tracing ----
        _dbg_groups_str = "[" + ", ".join(
            f"(idx={gi}, tangent={int(round(min(leading_edge_x(d) for d in grp)))})"
            if grp else f"(idx={gi}, tangent=None)"
            for gi, grp in enumerate(grouped_rows)
        ) + "]"
        _dbg_sel_idx: int | str = 0 if grouped_rows else "N/A"
        _dbg_sel_method = (
            "grouped_rows[0] (front-most group)"
            if grouped_rows
            else "grouped_rows[0] out of range (no groups detected), used []"
        )
        if self._row_advance_pending_log:
            self._row_advance_pending_log = False
            logger.info(
                "Row-SM POST-ADVANCE CHECK: active_row_index just became {} "
                "this frame's all_groups={} about to select using method='{}'",
                active_idx, _dbg_groups_str, _dbg_sel_method,
            )
        logger.info(
            "Row-SM SELECT DEBUG: active_row_index={} all_groups={} "
            "selected_group_list_index={} selection_method='{}' fresh_tangent={}",
            active_idx,
            _dbg_groups_str,
            _dbg_sel_idx,
            _dbg_sel_method,
            int(round(min_tang_x)) if min_tang_x is not None else "none",
        )
        # ---- END DIAGNOSTIC ----

        # Position memory fallback — applied to the active row's tracked position.
        max_mem = self._cfg.detection.max_memory_frames
        if min_tang_x is not None:
            self._last_known_row_leading_x = min_tang_x
            self._memory_frame_count = 0
            effective_tang_x = min_tang_x
            _using_fallback = False
        elif (
            self._last_known_row_leading_x is not None
            and self._memory_frame_count < max_mem
        ):
            self._memory_frame_count += 1
            effective_tang_x = self._last_known_row_leading_x
            _using_fallback = True
            logger.info(
                "Fallback row position: {}px (frame {}/{})",
                int(self._last_known_row_leading_x),
                self._memory_frame_count,
                max_mem,
            )
        else:
            effective_tang_x = None
            _using_fallback = False

        # Update display state (consumed by display block every frame).
        self._cycle_front_row = active_row
        self._cycle_row2 = row2
        self._cycle_ref_tangent_x = effective_tang_x

        # Throttled state log (~5 Hz).
        _now_log = time.monotonic()
        if _now_log >= self._cycle_log_next_time:
            self._cycle_log_next_time = _now_log + 0.2
            logger.info(
                "Row-SM state: active_row={} status={} this_row_tangent={} "
                "transfer_x={:.0f} grid_rows={}",
                active_idx,
                self._row_status,
                f"{int(effective_tang_x)}" if effective_tang_x is not None else "none",
                transfer_x,
                grid_rows,
            )

        # --- Step 3: stop trigger for the active row ---
        if self._row_status[active_idx] == "WAITING":
            if effective_tang_x is not None and effective_tang_x <= transfer_x:
                self._row_status[active_idx] = "STOPPED"
                logger.info(
                    "Row-SM FIRE: row {} tangent {} <= transfer_x {:.0f} "
                    "-> StopTuchabzug | row_status={}",
                    active_idx + 1,
                    int(effective_tang_x),
                    transfer_x,
                    self._row_status,
                )
                _safety = self._cfg.detection.boundary_safety_margin_px
                _capped_bnd = min(effective_tang_x, transfer_x - _safety)
                self._committed_boundary_x = _capped_bnd
                logger.info(
                    "Committed boundary SET: X={} (row {} fired at tangent={}, "
                    "transfer_x={:.0f}, capped to transfer_x - {})",
                    int(round(_capped_bnd)), active_idx + 1,
                    int(effective_tang_x), transfer_x, _safety,
                )
                self._sm.handle_stop_tuchabzug_trigger()
                self._push_signals_to_opcua()
                self._request_plc_stop_pulse()

    def _apply_belt_fault_debounce(
        self, result: "PipelineResult", *, belt_roi: "np.ndarray"
    ) -> None:
        """Process one belt inspection result through the debounce / re-arm logic.

        Raises a belt fault via the SM after ``belt_fault_debounce_frames``
        consecutive fault-verdict frames.  Re-arms fault detection after
        ``belt_fault_clear_frames`` consecutive clean frames.  Call on every
        belt inspection result regardless of whether a fault is currently active.
        """
        fault_verdict = result.verdict.is_fault
        debounce_frames = self._cfg.inspection.belt_fault_debounce_frames
        clear_frames = self._cfg.inspection.belt_fault_clear_frames

        if fault_verdict:
            self._belt_clear_count = 0
            if self._belt_fault_armed:
                self._belt_fault_debounce_count += 1
                logger.debug(
                    "Belt fault debounce {}/{} ({})",
                    self._belt_fault_debounce_count, debounce_frames, result.verdict.value,
                )
                if self._belt_fault_debounce_count >= debounce_frames:
                    # FIX 5: everything below — defect image, event log,
                    # ext_error write — only ever runs here, past this
                    # `>= debounce_frames` confirmation gate. A single-frame or
                    # otherwise unconfirmed fault verdict never reaches any of
                    # this; debounce_frames consecutive fault verdicts are
                    # required (the `else` branch below resets the counter on
                    # any clean frame in between). FIX 7: `result.verdict.is_fault`
                    # (this whole `if fault_verdict:` branch) is only ever True
                    # for FAULT_ROW_FUSED — or legacy FAULT_UNKNOWN when
                    # inspection.unknown_is_fault=True — per FIX 6's
                    # _verdict_from_row, so info/orange verdicts and
                    # unconfirmed transients can never reach the defect log or
                    # PLC write below.
                    self._belt_fault_armed = False
                    self._belt_fault_debounce_count = 0
                    if self._sm.raise_belt_fault(result.fault_reason or result.verdict.value):
                        self._counts.inspected += 1
                        self._counts.defects += 1
                        self._inference_view.add_recent_defect(result.verdict.value)
                        img_path = self._save_defect_image(belt_roi, result)
                        self._event_log.append(Event(
                            event_type=EventType.FAULT_RAISED,
                            profile_name=self._cfg.app.active_profile,
                            fault_reason=result.fault_reason or result.verdict.value,
                            image_filename=str(img_path) if img_path else "",
                            state_before=State.TRACKING.value,
                            state_after=State.FAULT.value,
                        ))
                        self._inference_view.set_counts(self._counts)
                        self._push_signals_to_opcua()
                        # FIX 4: SM fault_active=True (raise_belt_fault, just
                        # above) and the PLC ext_error write happen back-to-
                        # back, right here — the only two ways the shared
                        # fault latch is SET, always together.
                        self._request_plc_set_error()
                        logger.info(
                            "Belt fault raised after {} frames: verdict={} reason={}",
                            debounce_frames, result.verdict.value, result.fault_reason,
                        )
        else:
            self._belt_fault_debounce_count = 0
            if not self._belt_fault_armed:
                self._belt_clear_count += 1
                logger.debug(
                    "Belt clear frames {}/{}", self._belt_clear_count, clear_frames,
                )
                if self._belt_clear_count >= clear_frames:
                    self._belt_fault_armed = True
                    self._belt_clear_count = 0
                    logger.debug("Belt fault detection re-armed after {} clear frames", clear_frames)

    def _on_fault_cleared(self) -> None:
        """Common bookkeeping once the SM has actually left FAULT.

        Called from two places (FIX 4): the demo self-clear path (clean belt
        frames, see the gating above) and the production operator-ack path
        (_on_plc_error_quit, after StateMachine.acknowledge_fault returns
        True). This method itself never touches ext_error — in the ack path
        the caller already cleared it (PLC + UI clear together); in the demo
        path there is no PLC to clear.
        """
        self._belt_fault_armed = True
        self._belt_clear_count = 0
        self._event_log.append(Event(
            event_type=EventType.FAULT_CLEARED,
            profile_name=self._cfg.app.active_profile,
            state_before=State.FAULT.value,
            state_after=State.WAITING.value,
        ))
        self._push_signals_to_opcua()

    def _save_defect_image(self, image: np.ndarray, result: PipelineResult) -> Path | None:
        defect_dir = Path(self._cfg.storage.defect_image_dir)
        defect_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        path = defect_dir / f"{stamp}-{result.verdict.value}.jpg"
        try:
            cv2.imwrite(str(path), image)
            return path
        except Exception:  # noqa: BLE001
            logger.exception("failed to write defect image {}", path)
            return None

    # ---------- buttons ----------

    def _on_tuchabzug_rising_row_phase(self) -> None:
        """Update row phase when TuchabzugRunning goes False → True.

        AT_LINE → TRANSFERRING: PLC restarted the cloth to hand the row to the
        belt.  Tripwire is suppressed from this point until the line clears.
        Belt fault debounce is reset so each transfer window starts clean.
        """
        if self._row_phase == RowPhase.AT_LINE:
            self._row_phase = RowPhase.TRANSFERRING
            self._transfer_start_time = time.monotonic()
            self._transfer_timeout_warned = False
            # Fresh debounce state for every transfer window.
            self._belt_fault_debounce_count = 0
            self._belt_fault_armed = True
            self._belt_clear_count = 0
            logger.info("Transfer started: TRANSFERRING, tripwire suppressed")

    def _on_tuchabzug_falling_row_phase(self) -> None:
        """Update row phase when TuchabzugRunning goes True → False.

        TRANSFERRING → AT_LINE if the line is still occupied (PLC stopped
        mid-transfer); TRANSFERRING → IDLE if the line already cleared.
        """
        if self._row_phase == RowPhase.TRANSFERRING:
            if self._tripwire_prev_stable:
                self._row_phase = RowPhase.AT_LINE
            else:
                self._row_phase = RowPhase.IDLE
                self._line_clear_debounce_count = 0
        if self._cfg.detection.use_row_grouping:
            logger.info(
                "Row-SM pulse reset: StopTuchabzug=False | "
                "active_row={} status={} (unchanged)",
                self._active_row_index,
                self._row_status,
            )

    def _on_simulate_pulse(self) -> None:
        if self._cfg.app.mode != "demo":
            return
        # Demo: raise TuchabzugRunning for one simulated cloth cycle, then drop it.
        self._sm.handle_tuchabzug_rising()
        self._on_tuchabzug_rising_row_phase()
        self._push_signals_to_opcua()
        QTimer.singleShot(
            max(50, self._cfg.inspection.delay_after_pull_ms),
            self._demo_finish_pulse,
        )

    def _demo_finish_pulse(self) -> None:
        self._sm.handle_tuchabzug_falling()
        self._on_tuchabzug_falling_row_phase()
        self._push_signals_to_opcua()

    def _on_force_toggle_tuchabzug(self) -> None:
        if self._cfg.app.mode != "demo":
            return
        snap = self._sm.snapshot()
        if snap.tuchabzug_running:
            self._sm.handle_tuchabzug_falling()
            self._on_tuchabzug_falling_row_phase()
        else:
            self._sm.handle_tuchabzug_rising()
            self._on_tuchabzug_rising_row_phase()
        self._push_signals_to_opcua()

    def _on_save_snapshot(self) -> None:
        frame = self._capture_view.latest_frame()
        if frame is None:
            return
        path = self._capture_view.pick_save_path()
        if not path:
            return
        try:
            cv2.imwrite(str(path), frame)
        except Exception:  # noqa: BLE001
            logger.exception("failed to write snapshot {}", path)

    # ---------- CALIBRATION TOOLING: raw-frame recording ----------
    # See io/recorder.py. Recording is an observer only — it has no effect on
    # detection, the tripwire, or the state machine.

    def _on_record_toggled(self, active: bool) -> None:
        if active:
            session_dir = self._recorder.start()
            self._recorder_status_timer.start()
            self._capture_view.set_recording_state(True, 0, str(session_dir))
        else:
            self._recorder.stop()
            self._recorder_status_timer.stop()
            self._capture_view.set_recording_state(
                False, self._recorder.saved_count, str(self._recorder.session_dir or "")
            )

    def _refresh_recording_status(self) -> None:
        if self._recorder.is_recording:
            self._capture_view.set_recording_state(
                True, self._recorder.saved_count, str(self._recorder.session_dir or "")
            )

    # ---------- CALIBRATION TOOLING: playback frame source ----------
    # See io/camera.py PlaybackCamera. Switching source stops/restarts the
    # camera grab loop but does not touch detection thresholds, the tripwire,
    # or the state machine — recorded frames run through the exact same
    # _on_frame pipeline as the live camera.

    def _on_frame_source_changed(self, source: str) -> None:
        if source != "playback":
            self._switch_to_live()
            return
        if self._cfg.app.mode == "production":
            QMessageBox.warning(
                self, self.tr("Playback"),
                self.tr(
                    "Playback is only available in Demo mode — a live PLC "
                    "must never receive StopTuchabzug/fault writes from "
                    "replayed calibration frames."
                ),
            )
            self._service_view.set_source_combo("live")
            return
        folder = Path(self._cfg.playback.folder) if self._cfg.playback.folder else None
        if not folder or not folder.exists():
            QMessageBox.warning(
                self, self.tr("Playback"),
                self.tr("Choose a recorded-frame folder first (Browse…)."),
            )
            self._service_view.set_source_combo("live")
            return
        self._switch_to_playback(folder)

    def _switch_to_playback(self, folder: Path) -> None:
        was_running = self._camera_running
        if was_running:
            self._camera.stop()
            self._camera_running = False
        self._camera = make_camera(
            source="playback",
            mock_image_dir=Path(self._cfg.mock_camera.image_dir),
            mock_fps=self._cfg.mock_camera.fps,
            playback_dir=folder,
            playback_fps=self._cfg.playback.fps,
            playback_loop=self._cfg.playback.loop,
        )
        self._playback_active = True
        self._playback_status_timer.start()
        if was_running:
            self._start_camera()
        self._refresh_status_bar()
        logger.info("Switched to PlaybackCamera: {}", folder)

    def _switch_to_live(self) -> None:
        if not self._playback_active:
            return
        was_running = self._camera_running
        if was_running:
            self._camera.stop()
            self._camera_running = False
        self._camera = make_camera(
            source=self._cfg.camera.source,
            mock_image_dir=Path(self._cfg.mock_camera.image_dir),
            mock_fps=self._cfg.mock_camera.fps,
            pixel_format=self._cfg.camera.pixel_format,
            basler_serial=self._cfg.camera.serial,
            transform=self._build_transform(),
            on_warning=self._on_camera_warning,
        )
        self._playback_active = False
        self._playback_status_timer.stop()
        self._service_view.set_playback_status("")
        if was_running:
            self._start_camera()
        self._refresh_status_bar()
        logger.info("Switched back to live camera source={}", self._cfg.camera.source)

    def _on_playback_folder_changed(self, folder: str) -> None:
        self._save_cfg()
        if self._playback_active:
            self._switch_to_playback(Path(folder))

    def _on_playback_loop_changed(self, loop: bool) -> None:
        self._save_cfg()
        if isinstance(self._camera, PlaybackCamera):
            self._camera.set_loop(loop)

    def _on_playback_play(self) -> None:
        if isinstance(self._camera, PlaybackCamera):
            self._camera.resume()

    def _on_playback_pause(self) -> None:
        if isinstance(self._camera, PlaybackCamera):
            self._camera.pause()

    def _on_playback_step(self) -> None:
        if isinstance(self._camera, PlaybackCamera):
            self._camera.pause()
            self._camera.step()

    def _refresh_playback_status(self) -> None:
        if isinstance(self._camera, PlaybackCamera):
            self._service_view.set_playback_status(
                self.tr("Frame {idx} / {total}{paused}").format(
                    idx=self._camera.current_index + 1,
                    total=self._camera.frame_count,
                    paused=self.tr(" (paused)") if self._camera.is_paused() else "",
                )
            )

    def _on_export_logs(self) -> None:
        out_dir = self._review_view.pick_export_dir()
        if not out_dir:
            return
        log_dir = Path(self._cfg.storage.log_dir)
        copied = 0
        for p in log_dir.glob("events-*.csv"):
            try:
                (out_dir / p.name).write_bytes(p.read_bytes())
                copied += 1
            except OSError as e:
                logger.warning("failed to copy {}: {}", p, e)
        logger.info("exported {} log files to {}", copied, out_dir)

    # ---------- SERVICE handlers ----------

    def _on_mode_changed(self, mode: str) -> None:
        if mode == self._cfg.app.mode:
            return
        before = self._cfg.app.mode
        self._cfg.app.mode = mode
        # CALIBRATION TOOLING safety: never leave playback engaged when
        # switching into production — a replayed frame must not be able to
        # drive a live PLC stop/fault write.
        if mode == "production" and self._playback_active:
            self._service_view.set_source_combo("live")
            self._switch_to_live()
        self._inference_view.set_mode(Mode(mode))
        self._event_log.append(Event(
            event_type=EventType.MODE_CHANGE,
            extra={"from": before, "to": mode},
        ))
        self._save_cfg()
        self._refresh_status_bar()

    def _on_language_changed(self, lang: str) -> None:
        install_translator(lang)  # type: ignore[arg-type]
        self._cfg.app.language = lang
        self._save_cfg()

    def _on_orientation_changed(self) -> None:
        transform = self._build_transform()
        # Both camera backends expose set_transform via duck typing.
        if hasattr(self._camera, "set_transform"):
            self._camera.set_transform(transform)
        self._save_cfg()

    def _on_profile_changed(self, name: str) -> None:
        if not self._cfg.profile_store().has(name):
            return
        before = self._cfg.app.active_profile
        self._cfg.app.active_profile = name
        self._refresh_profile_cache()
        if self._camera_running:
            self._camera.apply_profile(self._active_profile_cache)
        self._event_log.append(Event(
            event_type=EventType.PROFILE_CHANGE,
            profile_name=name,
            extra={"from": before, "to": name},
        ))
        self._inference_view.set_profiles(
            self._cfg.profile_store().names(), self._cfg.app.active_profile,
        )
        self._save_cfg()
        self._refresh_status_bar()

    def _on_profile_renamed(self, old: str, new: str) -> None:
        store = self._cfg.profile_store()
        if not store.has(old) or store.has(new):
            return
        p = store.get(old).model_copy(update={"name": new})
        store.remove(old)
        store.upsert(p)
        if self._cfg.app.active_profile == old:
            self._cfg.app.active_profile = new
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()
        self._inference_view.set_profiles(store.names(), self._cfg.app.active_profile)

    def _on_profile_deleted(self, name: str) -> None:
        store = self._cfg.profile_store()
        if not store.has(name):
            return
        store.remove(name)
        if self._cfg.app.active_profile == name:
            self._cfg.app.active_profile = store.names()[0]
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()
        self._inference_view.set_profiles(store.names(), self._cfg.app.active_profile)

    def _on_web_settings_changed(self) -> None:
        self._save_cfg()
        self._refresh_status_bar()

    def _on_learn_reference(self, target_name: str | None = None) -> None:
        if self._latest_frame is None:
            return
        profile = self._active_profile_cache
        _belt, cloth_full = self._pipeline.split_rois(self._latest_frame, profile)
        # FIX 1: restrict calibration sampling to the cloth ROI — without this,
        # clutter below the cloth (rag, motor) could be sampled as fake
        # "single" pieces and pollute the learned geometry/shape stats.
        cloth, _cloth_rect = self._pipeline.apply_crop(cloth_full, profile.cloth_crop)

        # Bug fix: use the SAME detection method currently active
        # (detection.method), so Learn Reference samples exactly what the
        # calibration page's live preview is showing — not always blob.
        bg_reference = None
        if (
            self._cfg.detection.method == "bg_subtract"
            and self._bg_reference_cloth is not None
            and self._bg_reference_cloth.shape[:2] == cloth_full.shape[:2]
        ):
            bg_reference, _ = self._pipeline.apply_crop(self._bg_reference_cloth, profile.cloth_crop)
        detections = self._pipeline.detect_cloth_pieces(
            self._detector, cloth, profile, self._cfg.detection, bg_reference=bg_reference,
        )

        try:
            result = learn_reference(
                detections, base_profile=profile, new_name=target_name or None,
            )
        except ValueError as e:
            self._wizard_view.set_learn_result(self.tr("Learn failed: {err}").format(err=str(e)))
            return
        store = self._cfg.profile_store()
        store.upsert(result.profile)
        if target_name and self._cfg.app.active_profile != target_name:
            self._cfg.app.active_profile = target_name
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._inference_view.set_profiles(store.names(), self._cfg.app.active_profile)
        logger.info(
            "Learned shape: circularity={:.3f} (min {:.3f}), solidity={:.3f} (min {:.3f})",
            result.circularity_mean, result.circularity_min,
            result.solidity_mean, result.solidity_min,
        )
        msg = self.tr(
            "Learned: width={w}, height={h}, area={a}. "
            "Shape: circularity={cm:.3f} (min {cn:.3f}), solidity={sm:.3f} (min {sn:.3f}). "
            "Saved to profile [{name}]."
        ).format(
            w=result.width_px, h=result.height_px, a=result.area_px,
            cm=result.circularity_mean, cn=result.circularity_min,
            sm=result.solidity_mean, sn=result.solidity_min,
            name=result.profile.name,
        )
        self._wizard_view.set_learn_result(msg)
        self._event_log.append(Event(
            event_type=EventType.CALIBRATION_DONE,
            profile_name=result.profile.name,
            extra={
                "width_px": result.width_px,
                "height_px": result.height_px,
                "area_px": result.area_px,
                "samples": result.sample_count,
                "circularity_mean": result.circularity_mean,
                "circularity_min": result.circularity_min,
                "solidity_mean": result.solidity_mean,
                "solidity_min": result.solidity_min,
            },
        ))

    def _open_wizard(self, *, start_step: int = 0) -> None:
        self._wizard_active = True
        self._wizard_cfg_snapshot = copy.deepcopy(self._cfg)
        self._wizard_view.restart()
        self._wizard_view.set_known_profiles(self._cfg.profile_store().names())
        self._wizard_view.set_initial_orientation(
            self._cfg.orientation.rotation,
            self._cfg.orientation.flip_horizontal,
        )
        self._wizard_view.set_profile_values(self._active_profile_cache)
        # FIX 2: seed the calibration page's method combo/label + bg_subtract
        # reference from the same shared config the Service page reads.
        self._wizard_view.set_detection_settings(self._cfg.detection)
        self._wizard_view.set_bg_reference(self._bg_reference_cloth)
        self._wizard_view.set_cloth_reference(self._cloth_reference_mask)
        if start_step > 0:
            self._wizard_view.go_to_step(start_step)
        self._show_view("wizard")

    def _on_wizard_cancelled(self) -> None:
        """Discard all wizard draft changes and restore the original config."""
        self._wizard_active = False
        if self._wizard_cfg_snapshot is not None:
            self._cfg = self._wizard_cfg_snapshot
            self._wizard_cfg_snapshot = None
        self._refresh_profile_cache()
        if self._camera_running:
            self._camera.apply_profile(self._active_profile_cache)
            if hasattr(self._camera, "set_transform"):
                self._camera.set_transform(self._build_transform())
        self._show_view("inference")

    def _on_wizard_apply_orientation(self, rotation: int, flip: bool) -> None:
        self._cfg.orientation.rotation = rotation  # type: ignore[assignment]
        self._cfg.orientation.flip_horizontal = flip
        self._on_orientation_changed()

    def _on_wizard_roi_changed(self, x: int) -> None:
        # x is a FULL-FRAME column (guaranteed by WizardView._on_roi_dragged).
        assert x > 0, f"roi_split_x must be a positive full-frame column, got {x}"
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"roi_split_x": x})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        logger.info("Wizard step 2 Next: saving roi_split_x = {}, writing to profile {}", x, profile.name)
        self._save_cfg()
        self._refresh_profile_cache()
        logger.info("Profile cache updated: roi_split_x is now {}", self._active_profile_cache.roi_split_x)

    def _on_wizard_tl_changed(self, x: int) -> None:
        # x is a FULL-FRAME column (WizardView._on_tl_dragged adds roi_split_x).
        split_x = self._active_profile_cache.roi_split_x
        assert x > split_x, (
            f"transfer_line_x ({x}) must be right of roi_split_x ({split_x})"
        )
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"transfer_line_x": x})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        logger.info(
            "Wizard step 3 Next: saving transfer_line_x = {}, checking against roi_split_x = {}",
            x, split_x,
        )
        self._save_cfg()
        self._refresh_profile_cache()

    def _on_wizard_exposure_changed(self, us: int) -> None:
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"camera_exposure_us": us})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()
        if self._camera_running and hasattr(self._camera, "apply_profile"):
            self._camera.apply_profile(updated)

    def _on_wizard_gain_changed(self, gain: float) -> None:
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"camera_gain": gain})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()
        if self._camera_running and hasattr(self._camera, "apply_profile"):
            self._camera.apply_profile(updated)

    def _on_wizard_noise_threshold_changed(self, nt: float) -> None:
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"noise_threshold": nt})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()

    def _on_dough_is_darker_changed(self, value: bool) -> None:
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"dough_is_darker": value})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()

    def _on_belt_dough_is_darker_changed(self, value: bool) -> None:
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"belt_dough_is_darker": value})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()

    def _on_detection_method_changed(self, method: str) -> None:
        """A/B-switch the cloth detection method (app-level, not per-profile).

        FIX 2: single shared setting — fired by EITHER the Service page combo
        or the Calibration page combo, and pushes the new value back to BOTH
        so they never drift out of sync.
        """
        self._cfg.detection.method = method
        self._save_cfg()
        self._service_view.set_detection_method(method)
        self._wizard_view.set_detection_settings(self._cfg.detection)
        logger.info("cloth detection method switched to: {}", method)

    def _on_wizard_bridge_width_changed(self, width: int) -> None:
        """SECTION 4: persist the transfer-bridge width to the active profile."""
        profile = self._active_profile_cache
        updated = profile.model_copy(update={"transfer_bridge_width_px": max(1, width)})
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()
        logger.info("transfer bridge width set to {} px (profile {})", width, profile.name)

    def _on_wizard_grid_columns_changed(self, columns: int) -> None:
        """Persist columns-per-row (app-level detection setting)."""
        self._cfg.detection.grid_columns = max(1, columns)
        self._save_cfg()
        self._row_line_tracker.reset()
        logger.info("grid columns/row set to {}", columns)

    def _on_wizard_grid_rows_changed(self, rows: int) -> None:
        """Persist number-of-rows on the cloth (app-level detection setting)."""
        self._cfg.detection.grid_rows = max(0, rows)
        self._save_cfg()
        self._active_row_index = 0
        self._row_status = ["WAITING"] * max(0, self._cfg.detection.grid_rows)
        self._committed_boundary_x = None
        logger.info("grid rows set to {}", rows)

    def _on_row_grouping_changed(self, enabled: bool) -> None:
        """Toggle the per-row-stop row-grouping mode (app-level, shared flag).

        Persists config.detection.use_row_grouping and pushes it back to the
        calibration page so the checkbox + on/off label stay in sync. Resets the
        row tracker so the new mode starts clean on the next crossing.
        """
        self._cfg.detection.use_row_grouping = enabled
        self._save_cfg()
        self._row_line_tracker.reset()
        self._row_group_stop_active = False
        self._row_lines = []
        self._row_count = 0
        self._active_row_index = 0
        self._row_status = ["WAITING"] * max(0, self._cfg.detection.grid_rows)
        self._committed_boundary_x = None
        self._cycle_partial_warned = False
        self._cycle_front_row = []
        self._cycle_row2 = []
        self._cycle_ref_tangent_x = None
        self._last_known_row_leading_x = None
        self._memory_frame_count = 0
        self._wizard_view.set_detection_settings(self._cfg.detection)
        logger.info("cloth row grouping (per-row stop): {}", "ON" if enabled else "OFF")

    def _on_fill_mask_holes_changed(self, value: bool) -> None:
        self._cfg.detection.fill_mask_holes = value
        self._save_cfg()

    def _on_capture_bg_reference(self) -> None:
        """Snapshot the current cloth ROI (with no dough on it) as the
        bg_subtract baseline. See ClassicalDetector.detect_bg_subtract."""
        if self._latest_frame is None:
            QMessageBox.warning(
                self, self.tr("Detection"), self.tr("No frame available yet.")
            )
            return
        profile = self._active_profile_cache
        try:
            _belt, cloth = self._pipeline.split_rois(self._latest_frame, profile)
        except ValueError:
            QMessageBox.warning(
                self, self.tr("Detection"), self.tr("Could not read the current frame.")
            )
            return
        ref_path = self._config_dir / "bg_reference_cloth.png"
        if not cv2.imwrite(str(ref_path), cloth):
            QMessageBox.warning(
                self, self.tr("Detection"), self.tr("Failed to save the reference image.")
            )
            return
        self._bg_reference_cloth = cloth.copy()
        self._cfg.detection.bg_subtract.reference_path = str(ref_path)
        self._save_cfg()
        self._service_view.set_bg_reference_status(
            self.tr("Reference captured {ts}").format(ts=datetime.now().strftime("%H:%M:%S"))
        )
        # FIX 2: keep the calibration page's bg_subtract preview in sync too.
        self._wizard_view.set_bg_reference(self._bg_reference_cloth)
        logger.info("bg_subtract reference captured: {}", ref_path)

    def _on_save_cloth_reference(self, brightness_threshold: int) -> None:
        """Persist the Cloth Reference calibration: a one-time snapshot of the
        bright-cloth-vs-dark-metal boundary, used by Hough's cloth gating
        (see ClassicalDetector.detect_hough / compute_cloth_region_mask).

        The cloth/camera don't move during operation, so — exactly like
        ``_on_capture_bg_reference`` above — this is captured once here and
        reused as-is at runtime rather than recomputed every frame.
        """
        if self._latest_frame is None:
            QMessageBox.warning(
                self, self.tr("Detection"), self.tr("No frame available yet.")
            )
            return
        profile = self._active_profile_cache
        try:
            _belt, cloth = self._pipeline.split_rois(self._latest_frame, profile)
        except ValueError:
            QMessageBox.warning(
                self, self.tr("Detection"), self.tr("Could not read the current frame.")
            )
            return
        mask = self._detector.compute_cloth_region_mask(
            cloth, brightness_threshold=brightness_threshold
        )
        ref_path = self._config_dir / "cloth_reference_mask.png"
        if not cv2.imwrite(str(ref_path), mask):
            QMessageBox.warning(
                self, self.tr("Detection"), self.tr("Failed to save the cloth reference.")
            )
            return
        self._cloth_reference_mask = mask
        self._cfg.detection.hough.cloth_reference_path = str(ref_path)
        self._cfg.detection.hough.cloth_brightness_threshold = brightness_threshold
        self._save_cfg()
        self._service_view.set_cloth_reference_status(
            self.tr("Cloth Reference saved {ts}").format(ts=datetime.now().strftime("%H:%M:%S"))
        )
        self._wizard_view.set_cloth_reference(self._cloth_reference_mask)
        logger.info(
            "cloth reference saved: {} (threshold={})", ref_path, brightness_threshold
        )

    def _on_wizard_belt_crop_changed(self, top: int, bottom: int, left: int, right: int) -> None:
        from viscontrol.core.profiles import CropRegion
        profile = self._active_profile_cache
        updated = profile.model_copy(update={
            "belt_crop": CropRegion(top=top, bottom=bottom, left=left, right=right)
        })
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()

    def _on_wizard_cloth_crop_changed(self, top: int, bottom: int, left: int, right: int) -> None:
        from viscontrol.core.profiles import CropRegion
        profile = self._active_profile_cache
        updated = profile.model_copy(update={
            "cloth_crop": CropRegion(top=top, bottom=bottom, left=left, right=right)
        })
        store = self._cfg.profile_store()
        store.upsert(updated)
        self._cfg.profiles = store.as_list()
        self._save_cfg()
        self._refresh_profile_cache()

    def _on_wizard_completed(self, settings: dict) -> None:
        self._wizard_active = False
        self._wizard_cfg_snapshot = None
        # Apply orientation settings.
        self._cfg.orientation.rotation = settings.get("rotation", self._cfg.orientation.rotation)
        self._cfg.orientation.flip_horizontal = settings.get(
            "flip_horizontal", self._cfg.orientation.flip_horizontal
        )
        self._on_orientation_changed()

        # All coordinate values arriving from the wizard are FULL-FRAME columns
        # (roi_split_x, transfer_line_x). Apply to the active profile.
        profile = self._active_profile_cache
        profile_updates: dict = {}
        for key, field in (
            ("roi_split_x", "roi_split_x"),
            ("transfer_line_x", "transfer_line_x"),
            ("exposure_us", "camera_exposure_us"),
            ("gain", "camera_gain"),
        ):
            if key in settings:
                profile_updates[field] = settings[key]
        if profile_updates:
            updated = profile.model_copy(update=profile_updates)
            store = self._cfg.profile_store()
            store.upsert(updated)
            self._cfg.profiles = store.as_list()
            if self._camera_running and hasattr(self._camera, "apply_profile"):
                self._camera.apply_profile(updated)

        logger.info(
            "Wizard Finish: writing profile to {}",
            self._config_dir / "local.yaml",
        )
        self._save_cfg()
        self._refresh_profile_cache()
        p = self._active_profile_cache
        logger.info(
            "Profile written successfully. local.yaml contents: roi_split_x={}, transfer_line_x={}",
            p.roi_split_x, p.transfer_line_x,
        )
        self._event_log.append(Event(
            event_type=EventType.WIZARD_COMPLETE,
            profile_name=self._cfg.app.active_profile,
            extra=settings,
        ))
        install_translator(self._cfg.app.language)  # type: ignore[arg-type]
        self._show_view("inference")
        self._sm.force_reset_to_waiting()

    # ---------- OPC UA ----------

    def _on_opcua_tuchabzug_change(self, value: bool) -> None:
        """Called from the OPC UA thread when the PLC flips TuchabzugRunning."""
        snap = self._sm.snapshot()
        was = snap.tuchabzug_running
        if value and not was:
            self._sm.handle_tuchabzug_rising()
            self._on_tuchabzug_rising_row_phase()
        elif (not value) and was:
            self._sm.handle_tuchabzug_falling()
            self._on_tuchabzug_falling_row_phase()
        self._push_signals_to_opcua()

    def _push_signals_to_opcua(self) -> None:
        if self._opcua is None:
            return
        snap = self._sm.snapshot()
        try:
            self._opcua.publish_outputs(
                stop_tuchabzug=snap.stop_tuchabzug,
                fault_active=snap.fault_active,
            )
        except Exception:  # noqa: BLE001
            logger.exception("opcua publish failed")

    # ---------- Production PLC client callbacks ----------

    def _on_plc_tuchabzug_change(self, value: bool) -> None:
        """Called from the PLC polling thread on ext_tuchabzug_status edge."""
        snap = self._sm.snapshot()
        was = snap.tuchabzug_running
        if value and not was:
            self._sm.handle_tuchabzug_rising()
            self._on_tuchabzug_rising_row_phase()
        elif (not value) and was:
            self._sm.handle_tuchabzug_falling()
            self._on_tuchabzug_falling_row_phase()
        self._push_signals_to_opcua()

    def _on_plc_error_quit(self, value: bool) -> None:
        """Called from the PLC polling thread on the rising edge of ext_error_quit.

        FIX 4 (single shared latched fault state, UI <-> PLC): this is the
        ONE place the shared fault latch clears in production — the PLC's
        ext_error and the UI/state-machine FaultActive flag clear together,
        immediately, every time. Only fired on the rising edge by PlcClient,
        so this always means "clear now". Mirrors how the latch is SET
        (_apply_belt_fault_debounce: SM fault_active=True, then
        _request_plc_set_error(), back-to-back — see FIX 5).
        """
        if not value:
            return
        self._request_plc_clear_error()
        if self._sm.acknowledge_fault():
            self._on_fault_cleared()

    # ---------- Production PLC command helpers ----------

    def _request_plc_stop_pulse(self) -> None:
        """Queue a stop pulse on the PLC client (production only, no-op otherwise)."""
        if self._plc is None or self._cfg.app.mode != "production":
            return
        try:
            self._plc.send_stop_pulse()
        except Exception:  # noqa: BLE001
            logger.exception("plc.send_stop_pulse failed")

    def _request_plc_set_error(self) -> None:
        """Queue writing fault_error_code to ext_error (production only)."""
        if self._plc is None or self._cfg.app.mode != "production":
            return
        try:
            self._plc.set_error(self._cfg.plc.fault_error_code)
        except Exception:  # noqa: BLE001
            logger.exception("plc.set_error failed")

    def _request_plc_clear_error(self) -> None:
        """Queue clearing ext_error = 0 (production only)."""
        if self._plc is None or self._cfg.app.mode != "production":
            return
        try:
            self._plc.clear_error()
        except Exception:  # noqa: BLE001
            logger.exception("plc.clear_error failed")

    # ---------- SM listener ----------

    def _on_sm_change(self, old: State, new: State) -> None:
        # Could happen on a background thread (state machine has no thread affinity).
        # Push UI updates via QTimer.singleShot to ensure they hit the GUI thread.
        if new == State.TRACKING:
            QTimer.singleShot(0, self._reset_tracking_session)
        QTimer.singleShot(0, lambda: self._refresh_state(new))

    def _reset_tracking_session(self) -> None:
        """Reset per-cycle detection state at the start of every TuchabzugRunning phase."""
        self._tripwire_prev_stable = False
        self._tripwire_candidate = False
        self._tripwire_debounce_count = 0
        self._line_clear_debounce_count = 0
        self._belt_fault_debounce_count = 0
        self._transfer_timeout_warned = False
        self._row_line_tracker.reset()
        self._row_group_stop_active = False
        self._row_lines = []
        self._row_count = 0
        self._current_row_centroids = []
        self._detection_band = None
        self._last_hough_time = 0.0       # run immediately on first eligible frame
        self._cached_cloth_detections = []
        self._cached_cloth_highlight = []
        self._cached_tripwire_occupied = False
        self._active_bridge_width_px = 0
        self._detection_zone_outer_x = None
        # USE_ROW_GROUPING: _active_row_index/_row_status state machine persists
        # across rising edges and is managed in the rising-edge handler (_on_frame).
        # handle_tuchabzug_falling() already cleared stop_tuchabzug on the falling
        # edge, so no manual clear is needed here.
        self._cycle_partial_warned = False
        self._cycle_front_row = []
        self._cycle_row2 = []
        self._cycle_ref_tangent_x = None
        self._cycle_log_next_time = 0.0  # log immediately on first Hough run
        self._last_known_row_leading_x = None  # clear per-cycle position memory
        self._memory_frame_count = 0
        self._clear_sticky_overlay()
        logger.debug("Tracking session reset — new cycle")

    def _clear_sticky_overlay(self) -> None:
        self._sticky_overlay_timer.stop()
        self._sticky_belt_detections = []
        self._sticky_belt_crop_rect = None

    def _refresh_state(self, _state: State) -> None:
        snap = self._sm.snapshot()
        self._update_banner_from_signals(snap)

    def _update_banner_from_signals(self, snap: object) -> None:
        """Drive the status banner from PLC signals: FAULT > RUNNING > STOPPED.

        SERVICE overrides everything when the SM is in that state.
        """
        from viscontrol.ui.theme import ACCENT_RED, SUCCESS_GREEN, TEXT_SECONDARY, WARNING_AMBER
        if self._sm.state == State.SERVICE:
            self._inference_view.set_machine_status_label("SERVICE", WARNING_AMBER)
        elif snap.fault_active:  # type: ignore[attr-defined]
            self._inference_view.set_machine_status_label("FAULT", ACCENT_RED)
        elif snap.tuchabzug_running:  # type: ignore[attr-defined]
            self._inference_view.set_machine_status_label("RUNNING", SUCCESS_GREEN)
        else:
            self._inference_view.set_machine_status_label("STOPPED", TEXT_SECONDARY)

    # ---------- helpers ----------

    def _build_transform(self) -> OrientationTransform:
        return OrientationTransform(
            rotation=self._cfg.orientation.rotation,  # type: ignore[arg-type]
            flip_horizontal=self._cfg.orientation.flip_horizontal,
        )

    def _save_cfg(self) -> None:
        if self._wizard_active:
            return  # defer writes until Finish; cancel will revert in-memory state
        try:
            save_config(self._cfg, self._config_dir)
        except OSError as e:
            logger.error("failed to save config: {}", e)

    def _refresh_status_bar(self) -> None:
        connected = bool(getattr(self._camera, "is_connected", False)) and self._camera_running
        backend = getattr(self._camera, "backend_name", "camera")
        self._status.set_camera_connected(connected, label=backend.capitalize())
        self._status.set_profile(self._cfg.app.active_profile)
        mode = self._cfg.app.mode
        self._sidebar.set_mode(mode.upper(), is_production=(mode == "production"))
        self._status.set_mode(mode.upper())
        if mode == "production":
            if self._plc is not None:
                self._status.set_opcua(bool(getattr(self._plc, "is_connected", False)))
            else:
                opcua_connected = bool(getattr(self._opcua, "is_running", False))
                self._status.set_opcua(opcua_connected)
        else:
            self._status.set_opcua(None)
        port = self._cfg.web.port
        scheme = "http"
        self._status.set_web_url(f"{scheme}://0.0.0.0:{port}")

    def _on_camera_warning(self, msg: str) -> None:
        logger.warning("camera: {}", msg)

    def _build_startup_banner(self) -> QFrame:
        banner = QFrame()
        banner.setObjectName("startupBanner")
        banner.setFixedHeight(36)

        layout = QHBoxLayout(banner)
        layout.setContentsMargins(16, 0, 12, 0)

        text = QLabel(
            f"System started at {datetime.now().strftime('%H:%M')}"
        )
        close = QLabel("  ×  ")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(text)
        layout.addStretch(1)
        layout.addWidget(close)

        # Insert at the very top of the right-side content column (above the stack).
        self._content_layout.insertWidget(0, banner)

        def _dismiss() -> None:
            banner.hide()
            banner.setFixedHeight(0)

        close.mousePressEvent = lambda _e: _dismiss()
        QTimer.singleShot(self._cfg.ui.startup_banner_seconds * 1000, _dismiss)
        return banner
