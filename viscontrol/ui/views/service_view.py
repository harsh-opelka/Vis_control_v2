"""SERVICE view — PIN-gated settings.

Default PIN is ``0000`` (hashed on first run). After unlock the view shows
sections for:

- Mode (Demo / Production)
- Language (English / German) — NOT password-gated per spec, but still under
  SERVICE per spec; toggled here for convenience
- Orientation (rotation + horizontal flip)
- Profiles (rename / delete / set active)
- Web access (enable + set password)
- Change PIN
- Open Installation Wizard

The widget is "stateful" against the live :class:`AppConfig` — the main window
re-applies changed sections via signals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from viscontrol.core.config import AppConfig
from viscontrol.core.security import hash_pin, verify_pin
from viscontrol.ui.theme import (
    BORDER,
    FONT_LARGE,
    FONT_NORMAL,
    FONT_SMALL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from viscontrol.ui.widgets.camera_view import FramePreview


class ServiceView(QWidget):
    """PIN-gated settings panel.

    Signals (all "applied immediately" semantics):
      - mode_changed("demo"|"production")
      - language_changed("en"|"de")
      - orientation_changed()
      - profile_active_changed(name)
      - profile_renamed(old, new) / profile_deleted(name)
      - web_settings_changed()
      - pin_changed()
      - learn_reference_requested()
      - open_wizard_requested()
    """

    mode_changed = Signal(str)
    language_changed = Signal(str)
    orientation_changed = Signal()
    profile_active_changed = Signal(str)
    profile_renamed = Signal(str, str)
    profile_deleted = Signal(str)
    web_settings_changed = Signal()
    pin_changed = Signal()
    learn_reference_requested = Signal()
    open_wizard_requested = Signal()
    open_crop_wizard_requested = Signal()
    entered_service = Signal()
    exited_service = Signal()
    dough_is_darker_changed = Signal(bool)
    belt_dough_is_darker_changed = Signal(bool)
    show_belt_mask_changed = Signal(bool)
    show_cloth_mask_changed = Signal(bool)
    belt_detection_enabled_changed = Signal(bool)
    # Cloth-side alternative detection methods (A/B comparison). See
    # core/config.py _DetectionSection and detection/pipeline.py run_cloth_tracking.
    detection_method_changed = Signal(str)   # "blob" | "contour_external" | "hough" | "bg_subtract"
    fill_mask_holes_changed = Signal(bool)   # blob-only
    capture_bg_reference_requested = Signal()  # bg_subtract: snapshot the empty cloth
    open_cloth_reference_requested = Signal()  # hough: opens the wizard's Cloth Reference step
    # CALIBRATION TOOLING — frame source switch (live camera vs. recorded
    # playback folder). See main_window's _switch_to_playback/_switch_to_live.
    frame_source_changed = Signal(str)        # "live" | "playback"
    playback_folder_changed = Signal(str)
    playback_loop_changed = Signal(bool)
    playback_play_requested = Signal()
    playback_pause_requested = Signal()
    playback_step_requested = Signal()

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = config
        self._unlocked = False

        self._stack_layout = QStackedLayout(self)
        self._stack_layout.addWidget(self._build_lock_page())
        self._stack_layout.addWidget(self._build_settings_page())

    def attach_config(self, config: AppConfig) -> None:
        """Allow the main window to swap a fresh config in (e.g. after wizard)."""
        self._cfg = config
        self._populate_from_config()

    def lock(self) -> None:
        self._unlocked = False
        self._pin_input.clear()
        self._stack_layout.setCurrentIndex(0)
        self.exited_service.emit()

    # ---------- lock page ----------

    def _build_lock_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setObjectName("card")
        card.setMinimumWidth(360)
        card.setMaximumWidth(480)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(12)

        title = QLabel(self.tr("Service login"))
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_LARGE}pt; font-weight: 700;"
        )
        hint = QLabel(self.tr("Enter the SERVICE PIN. Default is 0000."))
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._pin_input = QLineEdit()
        self._pin_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._pin_input.setPlaceholderText(self.tr("PIN"))
        self._pin_input.returnPressed.connect(self._attempt_unlock)
        unlock = QPushButton(self.tr("Unlock"))
        unlock.setObjectName("primary")
        unlock.clicked.connect(self._attempt_unlock)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self._pin_input)
        layout.addWidget(unlock)
        outer.addWidget(card)
        return page

    def _attempt_unlock(self) -> None:
        pin = self._pin_input.text()
        if verify_pin(pin, self._cfg.ui.service_pin_hash):
            self._unlocked = True
            self._populate_from_config()
            self._stack_layout.setCurrentIndex(1)
            self.entered_service.emit()
        else:
            QMessageBox.warning(self, self.tr("Service"), self.tr("Incorrect PIN."))

    # ---------- settings page ----------

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        # Scrollable area for all settings cards.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 8)
        layout.setSpacing(16)

        layout.addWidget(self._build_mode_section())
        layout.addWidget(self._build_orientation_section())
        layout.addWidget(self._build_detection_section())
        layout.addWidget(self._build_diagnostics_section())
        layout.addWidget(self._build_frame_source_section())
        layout.addWidget(self._build_profile_section())
        layout.addWidget(self._build_web_section())
        layout.addWidget(self._build_pin_section())
        layout.addStretch(1)

        scroll.setWidget(content)
        page_layout.addWidget(scroll, 1)

        # Button row pinned outside the scroll area so it's always visible.
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(20, 8, 20, 16)
        btn_layout.addStretch(1)
        wizard_btn = QPushButton(self.tr("Open Installation Wizard"))
        wizard_btn.setObjectName("primary")
        wizard_btn.clicked.connect(self.open_wizard_requested.emit)
        btn_layout.addWidget(wizard_btn)
        crop_btn = QPushButton(self.tr("Adjust crops…"))
        crop_btn.setObjectName("secondary")
        crop_btn.clicked.connect(self.open_crop_wizard_requested.emit)
        btn_layout.addWidget(crop_btn)
        exit_btn = QPushButton(self.tr("Exit Service"))
        exit_btn.setObjectName("secondary")
        exit_btn.clicked.connect(self.lock)
        btn_layout.addWidget(exit_btn)
        page_layout.addWidget(btn_row)

        return page

    def _section_card(self, title: str) -> tuple[QFrame, QFormLayout]:
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 12, 20, 16)
        outer.setSpacing(10)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        outer.addWidget(title_lbl)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(8)
        outer.addLayout(form)
        return card, form

    def _build_mode_section(self) -> QWidget:
        card, form = self._section_card(self.tr("Mode and Language"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["demo", "production"])
        self._mode_combo.currentTextChanged.connect(
            lambda s: (self._cfg.app.__setattr__("mode", s), self.mode_changed.emit(s))
        )
        form.addRow(self.tr("Mode"), self._mode_combo)

        self._lang_combo = QComboBox()
        self._lang_combo.addItem(self.tr("English"), "en")
        self._lang_combo.addItem(self.tr("German (Deutsch)"), "de")
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        form.addRow(self.tr("Language"), self._lang_combo)

        self._lang_restart_notice = QLabel("")
        self._lang_restart_notice.setStyleSheet(
            f"color: #D9941A; font-size: {FONT_SMALL}pt;"
        )
        self._lang_restart_notice.setWordWrap(True)
        self._lang_restart_notice.hide()
        form.addRow("", self._lang_restart_notice)
        return card

    def _build_orientation_section(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 12, 20, 16)
        outer.setSpacing(10)

        title_lbl = QLabel(self.tr("Camera orientation"))
        title_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        outer.addWidget(title_lbl)

        # Preview on the left, controls on the right.
        row = QHBoxLayout()
        row.setSpacing(20)

        self._orient_preview = FramePreview(width=280, height=210)
        row.addWidget(self._orient_preview)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(8)

        self._rotation_combo = QComboBox()
        for v in (0, 90, 180, 270):
            self._rotation_combo.addItem(f"{v}°", v)
        self._rotation_combo.currentIndexChanged.connect(self._on_orientation_changed)
        form.addRow(self.tr("Rotation"), self._rotation_combo)

        self._flip_check = QCheckBox(self.tr("Flip horizontally"))
        self._flip_check.toggled.connect(self._on_orientation_changed)
        form.addRow("", self._flip_check)

        learn_btn = QPushButton(self.tr("Learn Reference"))
        learn_btn.setObjectName("secondary")
        learn_btn.clicked.connect(self.learn_reference_requested.emit)
        form.addRow(self.tr("Calibration"), learn_btn)

        form_widget = QWidget()
        form_widget.setLayout(form)
        row.addWidget(form_widget)
        row.addStretch(1)

        outer.addLayout(row)
        return card

    def _build_detection_section(self) -> QWidget:
        card, form = self._section_card(self.tr("Detection"))

        self._svc_dough_darker_check = QCheckBox(
            self.tr("Cloth: dough darker than background (Teig dunkler als Hintergrund)")
        )
        self._svc_dough_darker_check.toggled.connect(self._on_svc_dough_darker_changed)
        form.addRow("", self._svc_dough_darker_check)
        cloth_hint = QLabel(
            self.tr(
                "Enable when dough appears darker than the Gärtuch. "
                "Disable for dark-background / IR back-lighting setups."
            )
        )
        cloth_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        cloth_hint.setWordWrap(True)
        form.addRow("", cloth_hint)

        self._svc_belt_darker_check = QCheckBox(
            self.tr("Belt: dough darker than belt background")
        )
        self._svc_belt_darker_check.toggled.connect(self._on_svc_belt_darker_changed)
        form.addRow("", self._svc_belt_darker_check)
        belt_hint = QLabel(
            self.tr(
                "Usually OFF: the belt is a dark wire-mesh and dough is lighter. "
                "Enable only if your belt is brighter than the dough pieces."
            )
        )
        belt_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        belt_hint.setWordWrap(True)
        form.addRow("", belt_hint)

        method_sep = QLabel(self.tr("Cloth detection method (A/B compare)"))
        method_sep.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt; font-weight: 600;"
        )
        form.addRow("", method_sep)

        self._detection_method_combo = QComboBox()
        self._detection_method_combo.addItem(self.tr("Blob (default)"), "blob")
        self._detection_method_combo.addItem(self.tr("Contour (external, no solidity)"), "contour_external")
        self._detection_method_combo.addItem(self.tr("Hough circles"), "hough")
        self._detection_method_combo.addItem(self.tr("Background subtraction"), "bg_subtract")
        self._detection_method_combo.currentIndexChanged.connect(self._on_detection_method_changed)
        form.addRow(self.tr("Method"), self._detection_method_combo)
        method_hint = QLabel(
            self.tr(
                "Belt detection always uses Blob; this only switches cloth tracking. "
                "Hough/contour/bg_subtract params are tunable in config/local.yaml "
                "under 'detection'."
            )
        )
        method_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        method_hint.setWordWrap(True)
        form.addRow("", method_hint)

        self._fill_mask_holes_check = QCheckBox(self.tr("Fill mask holes before extraction (Blob only)"))
        self._fill_mask_holes_check.setToolTip(
            self.tr(
                "Pre-fills enclosed holes in the threshold mask before blob "
                "extraction — recovers a solid piece that thresholded as a "
                "hollow ring (shiny dome highlight). Ignored by the other "
                "three methods."
            )
        )
        self._fill_mask_holes_check.toggled.connect(self.fill_mask_holes_changed.emit)
        form.addRow("", self._fill_mask_holes_check)

        bg_ref_row = QHBoxLayout()
        capture_bg_btn = QPushButton(self.tr("Capture empty-cloth reference"))
        capture_bg_btn.setObjectName("secondary")
        capture_bg_btn.setToolTip(
            self.tr(
                "bg_subtract only: snapshot the current cloth ROI (with no "
                "dough on it) as the baseline. Detection then reports "
                "whatever changed from this baseline."
            )
        )
        capture_bg_btn.clicked.connect(self.capture_bg_reference_requested.emit)
        self._bg_reference_status = QLabel("")
        self._bg_reference_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        bg_ref_row.addWidget(capture_bg_btn)
        bg_ref_row.addWidget(self._bg_reference_status)
        bg_ref_row.addStretch(1)
        form.addRow(self.tr("bg_subtract"), _row(bg_ref_row))

        cloth_ref_row = QHBoxLayout()
        open_cloth_ref_btn = QPushButton(self.tr("Set Cloth Reference…"))
        open_cloth_ref_btn.setObjectName("secondary")
        open_cloth_ref_btn.setToolTip(
            self.tr(
                "hough only: opens the Installation Wizard's Calibration step "
                "to set and visually verify the bright-cloth-vs-dark-metal "
                "threshold used to gate Hough circles to the cloth. One-time "
                "setup — the cloth/camera don't move during operation."
            )
        )
        open_cloth_ref_btn.clicked.connect(self.open_cloth_reference_requested.emit)
        self._cloth_reference_status = QLabel("")
        self._cloth_reference_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        cloth_ref_row.addWidget(open_cloth_ref_btn)
        cloth_ref_row.addWidget(self._cloth_reference_status)
        cloth_ref_row.addStretch(1)
        form.addRow(self.tr("hough"), _row(cloth_ref_row))

        return card

    def _on_svc_dough_darker_changed(self, checked: bool) -> None:
        self.dough_is_darker_changed.emit(checked)

    def _on_svc_belt_darker_changed(self, checked: bool) -> None:
        self.belt_dough_is_darker_changed.emit(checked)

    def _on_detection_method_changed(self) -> None:
        method = self._detection_method_combo.currentData()
        if method:
            self.detection_method_changed.emit(method)

    def set_bg_reference_status(self, text: str) -> None:
        """Called by MainWindow after a successful/failed reference capture."""
        self._bg_reference_status.setText(text)

    def set_cloth_reference_status(self, text: str) -> None:
        """Called by MainWindow after a successful Cloth Reference save."""
        self._cloth_reference_status.setText(text)

    def set_detection_method(self, method: str) -> None:
        """FIX 2: push a method change made on the Calibration page back onto
        this page's combo, without re-emitting detection_method_changed
        (avoids a signal ping-pong between the two pages).
        """
        idx = self._detection_method_combo.findData(method)
        if idx >= 0:
            self._detection_method_combo.blockSignals(True)
            self._detection_method_combo.setCurrentIndex(idx)
            self._detection_method_combo.blockSignals(False)

    def _build_diagnostics_section(self) -> QWidget:
        card, form = self._section_card(self.tr("Diagnostics"))
        self._belt_detection_check = QCheckBox(self.tr("Belt detection"))
        self._belt_detection_check.setToolTip(
            self.tr(
                "Debug toggle. Unchecking disables belt inspection entirely: "
                "the inspection window never opens, no belt-side detection or "
                "fault debounce runs, and belt processing is skipped in the "
                "frame loop. Cloth ROI detection and row-stop logic are "
                "unaffected. Takes effect immediately; an open window closes "
                "right away. Existing latched faults still require operator ack."
            )
        )
        self._belt_detection_check.toggled.connect(self.belt_detection_enabled_changed.emit)
        form.addRow("", self._belt_detection_check)

        self._belt_mask_check = QCheckBox(self.tr("Show belt detection mask"))
        self._belt_mask_check.setToolTip(
            self.tr(
                "Overlays the post-threshold binary mask thumbnail on the "
                "Belt camera view. Read-only — does not affect detection."
            )
        )
        self._belt_mask_check.toggled.connect(self.show_belt_mask_changed.emit)
        form.addRow("", self._belt_mask_check)

        self._cloth_mask_check = QCheckBox(self.tr("Show cloth detection mask"))
        self._cloth_mask_check.setToolTip(
            self.tr(
                "Overlays the active method's binary/diff mask thumbnail on "
                "the Cloth camera view. Read-only — does not affect detection."
            )
        )
        self._cloth_mask_check.toggled.connect(self.show_cloth_mask_changed.emit)
        form.addRow("", self._cloth_mask_check)
        return card

    def _build_frame_source_section(self) -> QWidget:
        """CALIBRATION TOOLING: switch between the live camera and a folder of
        recorded frames (see io/camera.py PlaybackCamera / io/recorder.py).
        Only available in Demo mode — playback never drives a live PLC.
        """
        card, form = self._section_card(self.tr("Frame source (calibration)"))

        hint = QLabel(self.tr(
            "Demo mode only. Recorded frames run through the exact same "
            "detection pipeline as the live camera — use this to tune "
            "thresholds offline without the machine running."
        ))
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hint.setWordWrap(True)
        form.addRow("", hint)

        self._source_combo = QComboBox()
        self._source_combo.addItem(self.tr("Live camera"), "live")
        self._source_combo.addItem(self.tr("Recorded folder (playback)"), "playback")
        self._source_combo.currentIndexChanged.connect(self._on_source_combo_changed)
        form.addRow(self.tr("Source"), self._source_combo)

        folder_row = QHBoxLayout()
        self._playback_folder_label = QLabel(self.tr("(none selected)"))
        self._playback_folder_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        browse_btn = QPushButton(self.tr("Browse…"))
        browse_btn.setObjectName("secondary")
        browse_btn.clicked.connect(self._on_browse_playback_folder)
        folder_row.addWidget(self._playback_folder_label, 1)
        folder_row.addWidget(browse_btn)
        form.addRow(self.tr("Folder"), _row(folder_row))

        self._playback_loop_check = QCheckBox(self.tr("Loop"))
        self._playback_loop_check.toggled.connect(self._on_playback_loop_changed)
        form.addRow("", self._playback_loop_check)

        transport_row = QHBoxLayout()
        play_btn = QPushButton(self.tr("Play"))
        play_btn.setObjectName("secondary")
        play_btn.clicked.connect(self.playback_play_requested.emit)
        pause_btn = QPushButton(self.tr("Pause"))
        pause_btn.setObjectName("secondary")
        pause_btn.clicked.connect(self.playback_pause_requested.emit)
        step_btn = QPushButton(self.tr("Step ▸"))
        step_btn.setObjectName("secondary")
        step_btn.clicked.connect(self.playback_step_requested.emit)
        transport_row.addWidget(play_btn)
        transport_row.addWidget(pause_btn)
        transport_row.addWidget(step_btn)
        transport_row.addStretch(1)
        form.addRow(self.tr("Playback"), _row(transport_row))

        self._playback_status_label = QLabel("")
        self._playback_status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        form.addRow("", self._playback_status_label)

        return card

    def _on_source_combo_changed(self) -> None:
        self.frame_source_changed.emit(self._source_combo.currentData())

    def _on_browse_playback_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select recorded-frame folder"),
            self._cfg.playback.folder or str(Path.cwd()),
        )
        if not folder:
            return
        self._cfg.playback.folder = folder
        self._playback_folder_label.setText(folder)
        self.playback_folder_changed.emit(folder)

    def _on_playback_loop_changed(self, checked: bool) -> None:
        self._cfg.playback.loop = checked
        self.playback_loop_changed.emit(checked)

    def set_source_combo(self, source: str) -> None:
        """Let main_window force the combo back to "live" if a switch is rejected."""
        idx = self._source_combo.findData(source)
        if idx >= 0:
            self._source_combo.blockSignals(True)
            self._source_combo.setCurrentIndex(idx)
            self._source_combo.blockSignals(False)

    def set_playback_status(self, text: str) -> None:
        self._playback_status_label.setText(text)

    def _build_profile_section(self) -> QWidget:
        card, form = self._section_card(self.tr("Profiles"))
        self._profile_list = QListWidget()
        self._profile_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {BORDER}; }}"
        )
        self._profile_list.setMaximumHeight(140)
        form.addRow(self.tr("Available"), self._profile_list)

        btns = QHBoxLayout()
        set_active_btn = QPushButton(self.tr("Set active"))
        set_active_btn.setObjectName("secondary")
        set_active_btn.clicked.connect(self._on_set_active)
        rename_btn = QPushButton(self.tr("Rename"))
        rename_btn.setObjectName("secondary")
        rename_btn.clicked.connect(self._on_rename_profile)
        delete_btn = QPushButton(self.tr("Delete"))
        delete_btn.setObjectName("secondary")
        delete_btn.clicked.connect(self._on_delete_profile)
        btns.addWidget(set_active_btn)
        btns.addWidget(rename_btn)
        btns.addWidget(delete_btn)
        btns.addStretch(1)
        form.addRow("", _row(btns))
        return card

    def _build_web_section(self) -> QWidget:
        card, form = self._section_card(self.tr("Web access"))
        self._web_enabled = QCheckBox(self.tr("Enable web dashboard"))
        self._web_enabled.toggled.connect(self._on_web_changed)
        form.addRow("", self._web_enabled)

        self._web_port = QSpinBox()
        self._web_port.setRange(1, 65535)
        self._web_port.valueChanged.connect(self._on_web_changed)
        form.addRow(self.tr("Port"), self._web_port)

        self._web_pw_btn = QPushButton(self.tr("Set web password…"))
        self._web_pw_btn.setObjectName("secondary")
        self._web_pw_btn.clicked.connect(self._on_set_web_password)
        form.addRow(self.tr("Password"), self._web_pw_btn)

        self._web_pw_status = QLabel("")
        self._web_pw_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        form.addRow("", self._web_pw_status)
        return card

    def _build_pin_section(self) -> QWidget:
        card, form = self._section_card(self.tr("Service PIN"))
        btn = QPushButton(self.tr("Change PIN…"))
        btn.setObjectName("secondary")
        btn.clicked.connect(self._on_change_pin)
        form.addRow("", btn)
        return card

    def set_preview_frame(self, frame: np.ndarray) -> None:
        """Push a camera frame to the orientation live-preview widget."""
        if self._unlocked:
            self._orient_preview.set_frame(frame)

    # ---------- populate from config ----------

    def _populate_from_config(self) -> None:
        cfg = self._cfg
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentText(cfg.app.mode)
        self._mode_combo.blockSignals(False)

        self._lang_combo.blockSignals(True)
        idx = self._lang_combo.findData(cfg.app.language)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        self._lang_combo.blockSignals(False)

        self._rotation_combo.blockSignals(True)
        ridx = self._rotation_combo.findData(cfg.orientation.rotation)
        if ridx >= 0:
            self._rotation_combo.setCurrentIndex(ridx)
        self._rotation_combo.blockSignals(False)

        self._flip_check.blockSignals(True)
        self._flip_check.setChecked(cfg.orientation.flip_horizontal)
        self._flip_check.blockSignals(False)

        try:
            active_profile = cfg.active_profile()
            self._svc_dough_darker_check.blockSignals(True)
            self._svc_dough_darker_check.setChecked(active_profile.dough_is_darker)
            self._svc_dough_darker_check.blockSignals(False)
            self._svc_belt_darker_check.blockSignals(True)
            self._svc_belt_darker_check.setChecked(active_profile.belt_dough_is_darker)
            self._svc_belt_darker_check.blockSignals(False)
        except Exception:
            pass

        self._detection_method_combo.blockSignals(True)
        midx = self._detection_method_combo.findData(cfg.detection.method)
        if midx >= 0:
            self._detection_method_combo.setCurrentIndex(midx)
        self._detection_method_combo.blockSignals(False)
        self._fill_mask_holes_check.blockSignals(True)
        self._fill_mask_holes_check.setChecked(cfg.detection.fill_mask_holes)
        self._fill_mask_holes_check.blockSignals(False)
        self._belt_detection_check.blockSignals(True)
        self._belt_detection_check.setChecked(cfg.inspection.belt_detection_enabled)
        self._belt_detection_check.blockSignals(False)
        self._bg_reference_status.setText(
            self.tr("Reference captured.") if cfg.detection.bg_subtract.reference_path
            else self.tr("No reference captured yet.")
        )
        self._cloth_reference_status.setText(
            self.tr("Cloth Reference set.") if cfg.detection.hough.cloth_reference_path
            else self.tr("Not set — using live fallback.")
        )

        self._profile_list.clear()
        active = cfg.app.active_profile
        for p in cfg.profile_store().as_list():
            suffix = "  ★" if p.name == active else ""
            self._profile_list.addItem(f"{p.name}{suffix}")

        self._web_enabled.blockSignals(True)
        self._web_enabled.setChecked(cfg.web.enabled)
        self._web_enabled.blockSignals(False)
        self._web_port.blockSignals(True)
        self._web_port.setValue(cfg.web.port)
        self._web_port.blockSignals(False)
        self._web_pw_status.setText(
            self.tr("Password set.") if cfg.web.password_hash else self.tr("No password set.")
        )

        # CALIBRATION TOOLING: frame source picker — does not reflect the
        # *runtime* playback toggle (that lives in main_window), only the
        # persisted folder/loop preference. Always starts at "Live" on a
        # fresh populate since playback is a session-only state.
        self._source_combo.blockSignals(True)
        self._source_combo.setCurrentIndex(0)
        self._source_combo.blockSignals(False)
        self._playback_folder_label.setText(cfg.playback.folder or self.tr("(none selected)"))
        self._playback_loop_check.blockSignals(True)
        self._playback_loop_check.setChecked(cfg.playback.loop)
        self._playback_loop_check.blockSignals(False)

    # ---------- handlers ----------

    def _on_lang_changed(self) -> None:
        lang = self._lang_combo.currentData()
        self._cfg.app.language = lang
        # Generate the message NOW (in the OLD language) before the translator is swapped.
        notice = self.tr("Language will be applied on next restart.")
        self._lang_restart_notice.setText(notice)
        self._lang_restart_notice.show()
        QTimer.singleShot(5000, self._lang_restart_notice.hide)
        self.language_changed.emit(lang)

    def _on_orientation_changed(self) -> None:
        self._cfg.orientation.rotation = self._rotation_combo.currentData()
        self._cfg.orientation.flip_horizontal = self._flip_check.isChecked()
        self.orientation_changed.emit()

    def _selected_profile_name(self) -> str | None:
        item = self._profile_list.currentItem()
        if not item:
            return None
        name = item.text().replace("  ★", "").strip()
        return name

    def _on_set_active(self) -> None:
        name = self._selected_profile_name()
        if not name:
            return
        self._cfg.app.active_profile = name
        self.profile_active_changed.emit(name)
        self._populate_from_config()

    def _on_rename_profile(self) -> None:
        old = self._selected_profile_name()
        if not old:
            return
        new, ok = QInputDialog.getText(
            self, self.tr("Rename profile"), self.tr("New name:"), text=old,
        )
        if not ok or not new.strip():
            return
        self.profile_renamed.emit(old, new.strip())
        self._populate_from_config()

    def _on_delete_profile(self) -> None:
        name = self._selected_profile_name()
        if not name:
            return
        if len(self._cfg.profiles) <= 1:
            QMessageBox.warning(
                self, self.tr("Service"),
                self.tr("At least one profile is required."),
            )
            return
        confirm = QMessageBox.question(
            self, self.tr("Delete profile"),
            self.tr("Delete profile '{name}'?").format(name=name),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.profile_deleted.emit(name)
        self._populate_from_config()

    def _on_web_changed(self) -> None:
        self._cfg.web.enabled = self._web_enabled.isChecked()
        self._cfg.web.port = self._web_port.value()
        self.web_settings_changed.emit()

    def _on_set_web_password(self) -> None:
        pw, ok = QInputDialog.getText(
            self, self.tr("Web password"),
            self.tr("New password (leave blank to disable):"),
            echo=QLineEdit.EchoMode.Password,
        )
        if not ok:
            return
        if pw:
            self._cfg.web.password_hash = hash_pin(pw)
        else:
            self._cfg.web.password_hash = ""
        self._web_pw_status.setText(
            self.tr("Password set.") if self._cfg.web.password_hash else self.tr("No password set.")
        )
        self.web_settings_changed.emit()

    def _on_change_pin(self) -> None:
        current, ok = QInputDialog.getText(
            self, self.tr("Current PIN"),
            self.tr("Enter current PIN:"),
            echo=QLineEdit.EchoMode.Password,
        )
        if not ok:
            return
        if not verify_pin(current, self._cfg.ui.service_pin_hash):
            QMessageBox.warning(self, self.tr("Service"), self.tr("Incorrect PIN."))
            return
        new, ok = QInputDialog.getText(
            self, self.tr("New PIN"),
            self.tr("Enter new PIN (4–8 digits):"),
            echo=QLineEdit.EchoMode.Password,
        )
        if not ok or not new:
            return
        self._cfg.ui.service_pin_hash = hash_pin(new)
        self.pin_changed.emit()
        QMessageBox.information(self, self.tr("Service"), self.tr("PIN changed."))


def _row(layout: QHBoxLayout) -> QWidget:
    """Convenience: wrap a layout in a QWidget for QFormLayout.addRow."""
    w = QWidget()
    w.setLayout(layout)
    return w
