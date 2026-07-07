"""The main "INFERENCE" view.

Layout mirrors the supplied mockup:

  +---------------------------------------------------------------------+
  |  profile ▼   [Inspected] [Good] [Defects]   defect rate   START      |
  |                                              + Simulate Pulse (demo) |
  |                                              + Recent Defects        |
  +---------------------------------------------------------------------+
  |                            <STATE>                                   |
  |                                                                      |
  |    [ Belt ROI camera ]               [ Cloth ROI camera ]            |
  |                                                                      |
  |    Inference: 12.3 ms                                                |
  +---------------------------------------------------------------------+
  |  PLC signals: [TuchabzugRunning] [StopTuchabzug] [FaultActive]       |
  |   (Demo only) [Force toggle TuchabzugRunning] [Simulate Pulse]       |
  +---------------------------------------------------------------------+

The view is "passive" — it doesn't own the camera or detector. The main
window wires it into the camera frame signal, the detection result signal,
and the state-machine change callback. Buttons emit signals back so the
main window can drive the SM / OPC UA / event log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from viscontrol.core.events import Mode, RowPhase, State
from viscontrol.core.logger import logger
from viscontrol.core.profiles import ProductProfile
from viscontrol.detection.base import Detection
from viscontrol.io.video_recorder import ViewVideoRecorder
from viscontrol.ui.theme import (
    ACCENT_RED,
    BORDER,
    FONT_NORMAL,
    FONT_SMALL,
    SUCCESS_GREEN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING_AMBER,
)
from viscontrol.ui.widgets.camera_view import CameraView, CameraViewState
from viscontrol.ui.widgets.recent_defects import RecentDefectsList
from viscontrol.ui.widgets.signal_pill import SignalPill
from viscontrol.ui.widgets.stat_card import StatCard
from viscontrol.ui.widgets.state_banner import StateBanner


@dataclass
class InferenceCounts:
    inspected: int = 0
    good: int = 0
    defects: int = 0

    @property
    def defect_rate(self) -> float:
        return (self.defects / self.inspected * 100.0) if self.inspected else 0.0


class InferenceView(QWidget):
    """Live inspection view."""

    # User actions — wired by the main window.
    start_clicked = Signal()
    stop_clicked = Signal()
    profile_changed = Signal(str)
    simulate_pulse_clicked = Signal()
    force_toggle_tuchabzug_clicked = Signal()
    einlaufband_toggled = Signal(bool)  # True = sim belt running, False = stopped
    # Layer 3 "Set Column Bands" calibration (staggered_layout): click near
    # each column's vertical center in the paused cloth view to record its Y.
    column_bands_mode_toggled = Signal(bool)  # True = entering calibration mode
    column_band_clicked = Signal(int)         # image-space Y, forwarded from the cloth view
    column_bands_confirm_clicked = Signal()
    column_bands_reset_clicked = Signal()

    # Processed-view recording target frame rate (independent of the detection
    # rate — the grab is throttled to this so files stay a reasonable size).
    RECORD_FPS = 15

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._counts = InferenceCounts()
        self._running = False
        self._mode = Mode.DEMO
        self._build()

        # --- Processed-view recording (background-thread MP4, see ViewVideoRecorder).
        # Records the capture container (both camera panels + overlays + PLC
        # signal pills) only; entirely separate from the detection pipeline.
        self._video_recorder = ViewVideoRecorder(Path("recordings"), fps=self.RECORD_FPS)
        self._rec_state = "idle"               # idle | recording | paused
        self._rec_accum = 0.0                  # accumulated active seconds
        self._rec_active_since: float | None = None
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(max(1, int(1000 / self.RECORD_FPS)))
        self._rec_timer.timeout.connect(self._on_rec_tick)
        self._update_rec_controls()

    # ---------- construction ----------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        root.addLayout(self._build_top_bar())
        root.addLayout(self._build_record_bar())

        # Everything inside _capture_container is what the recorder grabs: both
        # camera panels (with all overlays) + the PLC signal panel. The top bar
        # and the record controls themselves are deliberately left OUT of it.
        self._capture_container = QWidget()
        cap_layout = QVBoxLayout(self._capture_container)
        cap_layout.setContentsMargins(0, 0, 0, 0)
        cap_layout.setSpacing(8)
        cap_layout.addWidget(self._build_state_and_cameras(), 1)
        cap_layout.addWidget(self._build_plc_panel())
        root.addWidget(self._capture_container, 1)

    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(16)

        # Profile dropdown.
        profile_frame = QFrame()
        profile_frame.setObjectName("card")
        pf_layout = QVBoxLayout(profile_frame)
        pf_layout.setContentsMargins(12, 8, 12, 8)
        pf_layout.setSpacing(2)
        pf_label = QLabel(self.tr("Profile"))
        pf_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._profile_combo = QComboBox()
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)
        pf_layout.addWidget(pf_label)
        pf_layout.addWidget(self._profile_combo)
        bar.addWidget(profile_frame, 0)

        # Stat cards.
        self._card_inspected = StatCard(self.tr("Inspected"))
        self._card_good = StatCard(self.tr("Good"))
        self._card_defects = StatCard(self.tr("Defects"))
        bar.addWidget(self._card_inspected, 0)
        bar.addWidget(self._card_good, 0)
        bar.addWidget(self._card_defects, 0)

        # Defect rate label.
        self._defect_rate = QLabel(self.tr("Defect rate: 0.0%"))
        self._defect_rate.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; padding-left: 12px;"
        )
        bar.addWidget(self._defect_rate, 1)

        # Start / Stop big button.
        self._start_btn = QPushButton(self.tr("START"))
        self._start_btn.setObjectName("primary")
        self._start_btn.setMinimumHeight(60)
        self._start_btn.setMinimumWidth(140)
        self._start_btn.clicked.connect(self._on_start_clicked)
        bar.addWidget(self._start_btn, 0)

        # Demo-only simulate button (mirrored in the PLC panel below).
        self._sim_btn_top = QPushButton(self.tr("Simulate Pulse"))
        self._sim_btn_top.setObjectName("secondary")
        self._sim_btn_top.clicked.connect(self.simulate_pulse_clicked.emit)
        bar.addWidget(self._sim_btn_top, 0)

        # Recent defects panel.
        self._recent = RecentDefectsList(max_entries=20)
        self._recent.setMaximumWidth(240)
        bar.addWidget(self._recent, 0)

        return bar

    def _build_record_bar(self) -> QHBoxLayout:
        """Record / Pause / Stop controls + the REC/PAUSED indicator.

        Lives ABOVE the capture container, so the controls and indicator never
        appear in the recorded video — only the camera panels + PLC pills do.
        """
        bar = QHBoxLayout()
        bar.setSpacing(8)

        label = QLabel(self.tr("Recording:"))
        label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        bar.addWidget(label, 0)

        self._rec_record_btn = QPushButton(self.tr("● Record"))
        self._rec_record_btn.setObjectName("secondary")
        self._rec_record_btn.clicked.connect(self._on_record_clicked)
        bar.addWidget(self._rec_record_btn, 0)

        self._rec_pause_btn = QPushButton(self.tr("Pause"))
        self._rec_pause_btn.setObjectName("secondary")
        self._rec_pause_btn.clicked.connect(self._on_pause_clicked)
        bar.addWidget(self._rec_pause_btn, 0)

        self._rec_stop_btn = QPushButton(self.tr("Stop"))
        self._rec_stop_btn.setObjectName("secondary")
        self._rec_stop_btn.clicked.connect(self._on_stop_recording_clicked)
        bar.addWidget(self._rec_stop_btn, 0)

        # Live state indicator: red "● REC mm:ss" (blinking dot) / amber "❚❚ PAUSED".
        self._rec_indicator = QLabel("")
        self._rec_indicator.setMinimumWidth(150)
        self._rec_indicator.setStyleSheet(
            f"color: {ACCENT_RED}; font-size: {FONT_NORMAL}pt; font-weight: 700;"
        )
        bar.addWidget(self._rec_indicator, 0)

        bar.addStretch(1)

        # Last-saved path (shown after Stop).
        self._rec_path_label = QLabel("")
        self._rec_path_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;"
        )
        bar.addWidget(self._rec_path_label, 0)

        return bar

    def _build_state_and_cameras(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # Compact banner: uses FONT_LARGE (28pt) and fixed height 40px so the
        # camera row can claim the majority of the vertical space.
        self._state_banner = StateBanner(compact=True)
        self._state_banner.set_state(State.WAITING)
        layout.addWidget(self._state_banner, 0)

        cam_row = QHBoxLayout()
        cam_row.setSpacing(12)
        self._belt_view = CameraView(self.tr("Belt"))
        self._cloth_view = CameraView(self.tr("Cloth"))
        self._cloth_view.calibration_click.connect(self.column_band_clicked.emit)
        cam_row.addWidget(self._belt_view, 1)
        cam_row.addWidget(self._cloth_view, 1)
        layout.addLayout(cam_row, 1)

        # Layer 3 "Set Column Bands" calibration row: while paused, toggle
        # into click-capture mode, click each column's vertical center in the
        # cloth view above, then Confirm (derives bands as midpoints between
        # adjacent clicks) or Reset to start over. Hidden unless staggered
        # layout support is relevant — kept always-visible here since it's a
        # one-time setup action, not a runtime toggle.
        col_row = QHBoxLayout()
        col_row.setSpacing(8)
        self._column_bands_btn = QPushButton(self.tr("Set Column Bands"))
        self._column_bands_btn.setObjectName("secondary")
        self._column_bands_btn.setCheckable(True)
        self._column_bands_btn.toggled.connect(self.column_bands_mode_toggled.emit)
        col_row.addWidget(self._column_bands_btn, 0)
        self._column_bands_confirm_btn = QPushButton(self.tr("Confirm"))
        self._column_bands_confirm_btn.setObjectName("secondary")
        self._column_bands_confirm_btn.clicked.connect(self.column_bands_confirm_clicked.emit)
        col_row.addWidget(self._column_bands_confirm_btn, 0)
        self._column_bands_reset_btn = QPushButton(self.tr("Reset Clicks"))
        self._column_bands_reset_btn.setObjectName("secondary")
        self._column_bands_reset_btn.clicked.connect(self.column_bands_reset_clicked.emit)
        col_row.addWidget(self._column_bands_reset_btn, 0)
        self._column_bands_status = QLabel("")
        self._column_bands_status.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;"
        )
        col_row.addWidget(self._column_bands_status, 1)
        layout.addLayout(col_row, 0)

        self._inference_label = QLabel(self.tr("Inference: —"))
        self._inference_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;"
        )
        self._inference_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._inference_label, 0)

        return frame

    def _build_plc_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)

        title = QLabel(self.tr("PLC Signals"))
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt; font-weight: 600;"
        )
        layout.addWidget(title, 0, 0, 1, 4)

        # TuchabzugRunning: green when running (production "healthy" state).
        self._sig_tuchabzug = SignalPill(
            "TuchabzugRunning", healthy_when=True, glow_when_true=False
        )
        # StopTuchabzug: red glow when asserted (we *want* it to be FALSE in normal flow).
        self._sig_stop = SignalPill(
            "StopTuchabzug", healthy_when=False, glow_when_true=True
        )
        # FaultActive: red glow when asserted; green otherwise.
        self._sig_fault = SignalPill(
            "FaultActive", healthy_when=False, glow_when_true=True
        )
        # EinlaufbandRunning: green when the belt inspection window is open.
        # Reflects toext_Einlaufband_running from the PLC in production; falls
        # back to the manual toggle below in demo mode / when no PLC is attached.
        self._sig_einlaufband = SignalPill(
            "EinlaufbandRunning", healthy_when=True, glow_when_true=False
        )
        layout.addWidget(self._sig_tuchabzug, 1, 0)
        layout.addWidget(self._sig_stop, 1, 1)
        layout.addWidget(self._sig_fault, 1, 2)
        layout.addWidget(self._sig_einlaufband, 1, 3)

        # Einlaufband manual toggle — fallback/override source for the belt
        # inspection window when no PLC is attached (demo mode). FIX 3: the
        # real PLC node (ns=6;s=::Einlauf:toext_Einlaufband_running) is the
        # actual source in production (see MainWindow._read_belt_inspect_signal);
        # this row is demo-only UI and is hidden in production by set_mode().
        self._einl_row = QFrame()
        einl_row = QHBoxLayout(self._einl_row)
        einl_row.setContentsMargins(0, 0, 0, 0)
        einl_label = QLabel(self.tr("Belt sim:"))
        einl_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._einlaufband_btn = QPushButton(self.tr("Einlaufband Running (sim)"))
        self._einlaufband_btn.setObjectName("secondary")
        self._einlaufband_btn.setCheckable(True)
        self._einlaufband_btn.toggled.connect(self.einlaufband_toggled.emit)
        einl_row.addWidget(einl_label, 0)
        einl_row.addWidget(self._einlaufband_btn, 0)
        einl_row.addStretch(1)
        layout.addWidget(self._einl_row, 2, 0, 1, 4)

        # Demo-only force-toggle row.
        self._demo_row = QFrame()
        demo_layout = QHBoxLayout(self._demo_row)
        demo_layout.setContentsMargins(0, 0, 0, 0)

        demo_label = QLabel(self.tr("Demo controls:"))
        demo_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._force_toggle_btn = QPushButton(self.tr("Force toggle TuchabzugRunning"))
        self._force_toggle_btn.setObjectName("secondary")
        self._force_toggle_btn.clicked.connect(self.force_toggle_tuchabzug_clicked.emit)

        self._sim_btn_bottom = QPushButton(self.tr("Simulate Pulse"))
        self._sim_btn_bottom.setObjectName("secondary")
        self._sim_btn_bottom.clicked.connect(self.simulate_pulse_clicked.emit)

        demo_layout.addWidget(demo_label, 0)
        demo_layout.addWidget(self._force_toggle_btn, 0)
        demo_layout.addWidget(self._sim_btn_bottom, 0)
        demo_layout.addStretch(1)
        layout.addWidget(self._demo_row, 3, 0, 1, 4)

        return frame

    # ---------- handlers ----------

    def _on_start_clicked(self) -> None:
        self._running = not self._running
        if self._running:
            self._start_btn.setText(self.tr("STOP"))
            self._start_btn.setObjectName("stop")
            self.start_clicked.emit()
        else:
            self._start_btn.setText(self.tr("START"))
            self._start_btn.setObjectName("primary")
            self.stop_clicked.emit()
        # Re-apply the stylesheet so objectName change takes effect.
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)

    def _on_profile_changed(self, name: str) -> None:
        if name:
            self.profile_changed.emit(name)

    # ---------- public setters used by the main window ----------

    def set_profiles(self, names: list[str], active: str) -> None:
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        self._profile_combo.addItems(names)
        if active in names:
            self._profile_combo.setCurrentText(active)
        self._profile_combo.blockSignals(False)

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode
        is_demo = mode == Mode.DEMO
        self._sim_btn_top.setVisible(is_demo)
        self._demo_row.setVisible(is_demo)
        # FIX 3: the manual "Einlaufband Running (sim)" toggle is a demo-only
        # stand-in for the real PLC node — never shown in production.
        self._einl_row.setVisible(is_demo)

    def set_state(self, state: State) -> None:
        self._state_banner.set_state(state)

    def set_machine_status_label(self, label: str, color: str) -> None:
        """Update the banner with a signal-driven label (FAULT/RUNNING/STOPPED/SERVICE)."""
        self._state_banner.set_display(label, color)

    def set_counts(self, counts: InferenceCounts) -> None:
        self._counts = counts
        self._card_inspected.set_value(str(counts.inspected))
        self._card_good.set_value(str(counts.good))
        self._card_defects.set_value(str(counts.defects))
        self._defect_rate.setText(
            self.tr("Defect rate: {rate:.1f}%").format(rate=counts.defect_rate)
        )

    def set_belt_frame(
        self,
        image: np.ndarray | None,
        detections: list[Detection] | None = None,
        px_per_mm: float = 1.0,
        crop_rect: tuple[int, int, int, int] | None = None,
        debug_mask: np.ndarray | None = None,
    ) -> None:
        self._belt_view.set_state(
            CameraViewState(
                image=image,
                detections=detections or [],
                px_per_mm=px_per_mm,
                crop_rect=crop_rect,
                debug_mask=debug_mask,
            )
        )

    def set_cloth_frame(
        self,
        image: np.ndarray | None,
        *,
        detections: list[Detection] | None = None,
        transfer_line_local_x: int | None = None,
        highlight_centroids: list[tuple[float, float]] | None = None,
        px_per_mm: float = 1.0,
        crop_rect: tuple[int, int, int, int] | None = None,
        tripwire_half_width_px: int = 0,
        tripwire_occupied: bool = False,
        transfer_bridge_width_px: int = 0,
        row_profile: np.ndarray | None = None,
        row_profile_x_offset: int = 0,
        row_profile_scale: float = 1.0,
        row_lines: list[float] | None = None,
        row_count: int | None = None,
        detection_band: tuple[int, int] | None = None,
        current_row_centroids: list[tuple[float, float]] | None = None,
        detection_zone_outer_x: int | None = None,
        grid_row_assignments: dict[int, int] | None = None,
        grid_ref_tangent_x: float | None = None,
        grid_label: str | None = None,
        debug_mask: np.ndarray | None = None,
        column_band_clicks: list[int] | None = None,
    ) -> None:
        self._cloth_view.set_state(
            CameraViewState(
                image=image,
                detections=detections or [],
                transfer_line_local_x=transfer_line_local_x,
                highlight_centroids=highlight_centroids or [],
                px_per_mm=px_per_mm,
                crop_rect=crop_rect,
                tripwire_half_width_px=tripwire_half_width_px,
                tripwire_occupied=tripwire_occupied,
                transfer_bridge_width_px=transfer_bridge_width_px,
                # DIAGNOSTIC ONLY — see camera_view.CameraViewState.row_profile.
                row_profile=row_profile,
                row_profile_x_offset=row_profile_x_offset,
                row_profile_scale=row_profile_scale,
                row_lines=row_lines,
                row_count=row_count,
                detection_band=detection_band,
                current_row_centroids=current_row_centroids or [],
                detection_zone_outer_x=detection_zone_outer_x,
                grid_row_assignments=grid_row_assignments,
                grid_ref_tangent_x=grid_ref_tangent_x,
                grid_label=grid_label,
                debug_mask=debug_mask,
                column_band_clicks=column_band_clicks,
            )
        )

    def set_column_bands_calibration_mode(self, active: bool) -> None:
        """Layer 3: enter/exit Set Column Bands click-capture mode.

        Syncs the toggle button's checked state (without re-emitting
        column_bands_mode_toggled) and enables click capture on the cloth view.
        """
        self._cloth_view.set_calibration_mode(active)
        self._column_bands_btn.blockSignals(True)
        self._column_bands_btn.setChecked(active)
        self._column_bands_btn.blockSignals(False)

    def set_column_bands_status(self, text: str) -> None:
        self._column_bands_status.setText(text)

    def set_inference_ms(self, ms: float) -> None:
        self._inference_label.setText(self.tr("Inference: {ms:.1f} ms").format(ms=ms))

    def set_plc_signals(
        self,
        *,
        tuchabzug_running: bool,
        stop_tuchabzug: bool,
        fault_active: bool,
        einlaufband_running: bool = False,
    ) -> None:
        self._sig_tuchabzug.set_value(tuchabzug_running)
        self._sig_stop.set_value(stop_tuchabzug)
        self._sig_fault.set_value(fault_active)
        self._sig_einlaufband.set_value(einlaufband_running)

    def set_belt_inspect_state(self, window_open: bool) -> None:
        """Update the belt ROI state label above the camera image."""
        self._belt_view.set_state_label(
            "INSPECTING" if window_open else "IDLE",
            active=window_open,
        )

    def set_cloth_row_phase(self, phase: RowPhase) -> None:
        """Update the cloth ROI state label above the camera image."""
        active = phase != RowPhase.IDLE
        self._cloth_view.set_state_label(phase.value, active=active)

    def add_recent_defect(self, reason: str) -> None:
        self._recent.add_entry(reason)

    # ---------- processed-view recording ----------

    def _grab_frame(self) -> np.ndarray | None:
        """Render the capture container (both panels + overlays + pills) to a
        BGR uint8 array on the GUI thread. Cheap — it blits the already-painted
        widget; the encoding happens on the recorder's background thread."""
        pixmap = self._capture_container.grab()
        if pixmap.isNull():
            return None
        img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        if w == 0 or h == 0:
            return None
        bpl = img.bytesPerLine()  # row stride (padded), may exceed w*3
        buf = img.constBits()
        arr = np.frombuffer(buf, dtype=np.uint8)[: bpl * h].reshape(h, bpl)
        arr = arr[:, : w * 3].reshape(h, w, 3)
        return arr[:, :, ::-1].copy()  # RGB → BGR for OpenCV

    def _on_record_clicked(self) -> None:
        if self._rec_state == "recording":
            return
        if self._rec_state == "paused":
            # Resume: keep the same file, just start appending again.
            self._rec_state = "recording"
            self._rec_active_since = time.monotonic()
            self._update_rec_controls()
            return
        # Start a fresh recording.
        frame = self._grab_frame()
        if frame is None:
            logger.error("recording: could not grab the inference view")
            self._rec_path_label.setText(self.tr("Recording: could not capture view"))
            return
        h, w = frame.shape[:2]
        try:
            path = self._video_recorder.start(w, h)
        except Exception:  # noqa: BLE001
            logger.exception("recording: failed to start")
            path = None
        if path is None:
            self._rec_path_label.setText(self.tr("Recording failed to start (codec?)"))
            return
        self._rec_state = "recording"
        self._rec_accum = 0.0
        self._rec_active_since = time.monotonic()
        self._rec_path_label.setText("")
        self._video_recorder.submit(frame)  # first frame
        if not self._rec_timer.isActive():
            self._rec_timer.start()
        self._update_rec_controls()

    def _on_pause_clicked(self) -> None:
        if self._rec_state != "recording":
            return
        if self._rec_active_since is not None:
            self._rec_accum += time.monotonic() - self._rec_active_since
        self._rec_active_since = None
        self._rec_state = "paused"
        self._update_rec_controls()

    def _on_stop_recording_clicked(self) -> None:
        if self._rec_state == "idle":
            return
        self._rec_timer.stop()
        if self._rec_state == "recording" and self._rec_active_since is not None:
            self._rec_accum += time.monotonic() - self._rec_active_since
        self._rec_state = "idle"
        self._rec_active_since = None
        try:
            path = self._video_recorder.stop()
        except Exception:  # noqa: BLE001
            logger.exception("recording: failed to finalise")
            path = None
        self._rec_indicator.setText("")
        if path is not None:
            self._rec_path_label.setText(self.tr("Saved: {p}").format(p=str(path)))
            logger.info("recording saved: {}", path)
        self._update_rec_controls()

    def _on_rec_tick(self) -> None:
        """Grab+submit one frame while recording (skipped while paused), and
        refresh the REC/PAUSED indicator. Runs on the GUI thread at RECORD_FPS;
        the heavy encode work is on the recorder's background thread."""
        if self._rec_state == "recording":
            frame = self._grab_frame()
            if frame is not None:
                self._video_recorder.submit(frame)
        self._update_rec_indicator()

    def _update_rec_indicator(self) -> None:
        if self._rec_state == "idle":
            self._rec_indicator.setText("")
            return
        elapsed = self._rec_accum
        if self._rec_active_since is not None:
            elapsed += time.monotonic() - self._rec_active_since
        mm, ss = int(elapsed // 60), int(elapsed % 60)
        if self._rec_state == "paused":
            self._rec_indicator.setStyleSheet(
                f"color: {WARNING_AMBER}; font-size: {FONT_NORMAL}pt; font-weight: 700;"
            )
            self._rec_indicator.setText(self.tr("❚❚ PAUSED  {m:02d}:{s:02d}").format(m=mm, s=ss))
        else:
            # Blink the dot ~1 Hz so it's obvious recording is live.
            dot = "●" if int(elapsed * 2) % 2 == 0 else "  "
            self._rec_indicator.setStyleSheet(
                f"color: {ACCENT_RED}; font-size: {FONT_NORMAL}pt; font-weight: 700;"
            )
            self._rec_indicator.setText(
                self.tr("{d} REC  {m:02d}:{s:02d}").format(d=dot, m=mm, s=ss)
            )

    def _update_rec_controls(self) -> None:
        recording = self._rec_state == "recording"
        paused = self._rec_state == "paused"
        idle = self._rec_state == "idle"
        self._rec_record_btn.setText(self.tr("Resume") if paused else self.tr("● Record"))
        self._rec_record_btn.setEnabled(idle or paused)
        self._rec_pause_btn.setEnabled(recording)
        self._rec_stop_btn.setEnabled(not idle)

    def stop_recording_on_shutdown(self) -> None:
        """Finalise any in-progress recording on app close (called by MainWindow)."""
        if self._rec_state != "idle":
            self._on_stop_recording_clicked()

    def set_roi_info(
        self,
        roi_split_x: int,
        original_width: int,
        transfer_line_x: int,
    ) -> None:
        """Set coordinate-debug captions below each camera view.

        All three arguments are in FULL-FRAME (original camera image) coordinates.
        The caption under Belt shows the belt column range; the caption under Cloth
        shows the cloth column range and the transfer-line position.
        """
        self._belt_view.set_caption(
            f"Belt: col 0 – {roi_split_x - 1} px (orig)"
        )
        self._cloth_view.set_caption(
            f"Cloth: col {roi_split_x} – {original_width - 1} px (orig)"
            f"  ·  Transfer line: {transfer_line_x} px"
        )

    def reset_running_button(self) -> None:
        self._running = False
        self._start_btn.setText(self.tr("START"))
        self._start_btn.setObjectName("primary")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
