"""Simple stat card: label + big number."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from viscontrol.ui.theme import (
    CARD_RADIUS,
    FONT_BIG_NUMBER,
    FONT_SMALL,
    STAT_CARD_HEIGHT,
    STAT_CARD_WIDTH,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class StatCard(QFrame):
    """Card showing a small label above a large number."""

    def __init__(self, title: str, value: str = "0", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedSize(STAT_CARD_WIDTH, STAT_CARD_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        self._title = QLabel(title)
        self._title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._value = QLabel(value)
        self._value.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_BIG_NUMBER}pt; font-weight: 600;"
        )
        self._value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, value: str) -> None:
        self._value.setText(value)

    def set_title(self, title: str) -> None:
        self._title.setText(title)
