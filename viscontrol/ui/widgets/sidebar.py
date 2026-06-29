"""Left navigation sidebar."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from viscontrol.ui.theme import FONT_LARGE, FONT_SMALL, SIDEBAR_WIDTH


class Sidebar(QFrame):
    """Navy sidebar with VISCONTROL header, nav buttons, and a mode pill at the bottom."""

    nav_changed = Signal(str)
    mode_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(SIDEBAR_WIDTH)

        self._buttons: dict[str, QPushButton] = {}
        self._mode_btn: QPushButton | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 24, 0, 24)
        layout.setSpacing(4)

        header = QLabel("VISCONTROL")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 0 20px;")
        layout.addWidget(header)

        sub = QLabel("OPELKA")
        sub.setStyleSheet(f"font-size: {FONT_SMALL}pt; padding: 0 20px 24px 20px; opacity: 0.6;")
        layout.addWidget(sub)

        for nav_id, label in [
            ("capture", "CAPTURE"),
            ("inference", "INFERENCE"),
            ("review", "REVIEW"),
            ("service", "SERVICE"),
        ]:
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, n=nav_id: self.nav_changed.emit(n))
            layout.addWidget(btn)
            self._buttons[nav_id] = btn

        layout.addStretch(1)

        self._mode_btn = QPushButton("DEMO")
        self._mode_btn.setObjectName("modePill")
        self._mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_btn.clicked.connect(self.mode_clicked.emit)
        self._mode_btn.setStyleSheet(
            "QPushButton#modePill {"
            " background-color: #C8102E; color: white;"
            " border-radius: 14px; padding: 6px 18px;"
            " margin: 0 20px; font-weight: bold;"
            "}"
        )
        layout.addWidget(self._mode_btn)

    def set_active(self, nav_id: str) -> None:
        for k, btn in self._buttons.items():
            btn.setProperty("selected", k == nav_id)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def set_mode(self, mode_label: str, *, is_production: bool) -> None:
        if not self._mode_btn:
            return
        self._mode_btn.setText(mode_label)
        color = "#2D8659" if is_production else "#C8102E"
        self._mode_btn.setStyleSheet(
            f"QPushButton#modePill {{"
            f" background-color: {color}; color: white;"
            f" border-radius: 14px; padding: 6px 18px;"
            f" margin: 0 20px; font-weight: bold;"
            f"}}"
        )
