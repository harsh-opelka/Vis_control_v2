"""Bottom status bar.

Composed of small text segments (camera dot, profile, mode, OPC UA dot, web
URL, disk, timestamp). The view that owns the InferenceView updates these
in response to background signals.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QStatusBar, QWidget

from viscontrol.ui.theme import ACCENT_RED, FONT_SMALL, SUCCESS_GREEN, TEXT_SECONDARY


class StatusBar(QStatusBar):
    """Composite status bar with named segments."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._camera = QLabel("● Camera")
        self._profile = QLabel("Profile: —")
        self._mode = QLabel("Mode: —")
        self._opcua = QLabel("● OPC UA")
        self._web = QLabel("http://—")
        self._disk = QLabel("Disk: —")
        self._time = QLabel("—")

        for lbl in (self._camera, self._profile, self._mode, self._opcua,
                    self._web, self._disk, self._time):
            lbl.setStyleSheet(f"font-size: {FONT_SMALL}pt; color: {TEXT_SECONDARY};")

        self.addWidget(self._camera, 0)
        self.addWidget(_separator(), 0)
        self.addWidget(self._profile, 0)
        self.addWidget(_separator(), 0)
        self.addWidget(self._mode, 0)
        self.addWidget(_separator(), 0)
        self.addWidget(self._opcua, 0)
        self.addWidget(_separator(), 0)
        self.addWidget(self._web, 1)
        self.addPermanentWidget(self._disk, 0)
        self.addPermanentWidget(self._time, 0)

        # Tick the clock + disk usage every second.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._tick()

    def _tick(self) -> None:
        self._time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            usage = shutil.disk_usage(Path.cwd())
            free_gb = usage.free / (1024 ** 3)
            self._disk.setText(f"Disk: {free_gb:.1f} GB free")
        except OSError:
            self._disk.setText("Disk: ?")

    def set_camera_connected(self, connected: bool, *, label: str = "Camera") -> None:
        color = SUCCESS_GREEN if connected else ACCENT_RED
        self._camera.setText(f"● {label}")
        self._camera.setStyleSheet(f"font-size: {FONT_SMALL}pt; color: {color};")

    def set_profile(self, profile_name: str) -> None:
        self._profile.setText(f"Profile: {profile_name}")

    def set_mode(self, label: str) -> None:
        self._mode.setText(f"Mode: {label}")

    def set_opcua(self, connected: bool | None) -> None:
        """``None`` = Demo mode (hide indicator)."""
        if connected is None:
            self._opcua.setText("(OPC UA disabled in Demo)")
            self._opcua.setStyleSheet(f"font-size: {FONT_SMALL}pt; color: {TEXT_SECONDARY};")
            return
        color = SUCCESS_GREEN if connected else ACCENT_RED
        self._opcua.setText("● OPC UA")
        self._opcua.setStyleSheet(f"font-size: {FONT_SMALL}pt; color: {color};")

    def set_web_url(self, url: str) -> None:
        self._web.setText(url)


def _separator() -> QLabel:
    sep = QLabel("|")
    sep.setStyleSheet(f"color: {TEXT_SECONDARY}; padding: 0 6px;")
    sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return sep
