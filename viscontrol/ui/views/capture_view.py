"""CAPTURE view — single-shot camera preview + "Save snapshot" button.

Used for setup work where the operator wants to grab a still image (to share
with support, attach to a wizard step, etc.) without going through the full
inspection cycle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from viscontrol.ui.theme import FONT_NORMAL, FONT_SMALL, TEXT_PRIMARY, TEXT_SECONDARY
from viscontrol.ui.widgets.camera_view import CameraView, CameraViewState


class CaptureView(QWidget):
    """Live preview + snapshot button.

    Also hosts the CALIBRATION-TOOLING "Record" control: continuous raw-frame
    capture to disk (see ``io/recorder.py``). Recording is purely an
    observer — it has no effect on detection.
    """

    snapshot_requested = Signal()  # main window handles disk write
    record_toggled = Signal(bool)  # main window starts/stops the FrameRecorder

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._latest_frame: np.ndarray | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        title_frame = QFrame()
        title_frame.setObjectName("card")
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(20, 16, 20, 16)
        title = QLabel(self.tr("Capture"))
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        hint = QLabel(
            self.tr("Live camera preview. Use 'Save snapshot' to write the current frame to disk.")
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        title_layout.addWidget(title)
        title_layout.addWidget(hint)
        root.addWidget(title_frame, 0)

        self._preview = CameraView(self.tr("Full frame"))
        root.addWidget(self._preview, 1)

        bar = QHBoxLayout()
        self._record_btn = QPushButton(self.tr("● Record"))
        self._record_btn.setObjectName("secondary")
        self._record_btn.setCheckable(True)
        self._record_btn.setToolTip(
            self.tr(
                "CALIBRATION TOOLING: record raw camera frames to disk as "
                "lossless PNGs for offline playback. Does not affect detection."
            )
        )
        self._record_btn.toggled.connect(self.record_toggled.emit)
        bar.addWidget(self._record_btn)

        self._record_status = QLabel("")
        self._record_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        bar.addWidget(self._record_status)

        bar.addStretch(1)
        self._snapshot_btn = QPushButton(self.tr("Save snapshot"))
        self._snapshot_btn.setObjectName("primary")
        self._snapshot_btn.clicked.connect(self.snapshot_requested.emit)
        bar.addWidget(self._snapshot_btn)
        root.addLayout(bar)

    def set_frame(self, image: np.ndarray | None) -> None:
        self._latest_frame = image
        self._preview.set_state(CameraViewState(image=image))

    def latest_frame(self) -> np.ndarray | None:
        return self._latest_frame

    def set_recording_state(self, active: bool, frame_count: int = 0, folder: str = "") -> None:
        """Reflect the FrameRecorder's state — CALIBRATION TOOLING, see main_window."""
        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(active)
        self._record_btn.blockSignals(False)
        self._record_btn.setText(self.tr("■ Stop") if active else self.tr("● Record"))
        if active:
            self._record_status.setStyleSheet(
                "color: #D9341A; font-size: {}pt; font-weight: 600;".format(FONT_SMALL)
            )
            self._record_status.setText(
                self.tr("Recording — {n} frames saved to {dir}").format(n=frame_count, dir=folder)
            )
        else:
            self._record_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
            self._record_status.setText(
                self.tr("Stopped — {n} frames saved to {dir}").format(n=frame_count, dir=folder)
                if frame_count else ""
            )

    def pick_save_path(self) -> Path | None:
        """Open a save-as dialog. Returns ``None`` if the user cancelled."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save snapshot"),
            "snapshot.png",
            self.tr("PNG image (*.png);;JPEG image (*.jpg *.jpeg)"),
        )
        return Path(path) if path else None
