"""Recent Defects list — most-recent first, capped by config."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from viscontrol.ui.theme import (
    ACCENT_RED,
    BORDER,
    FONT_NORMAL,
    FONT_SMALL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class RecentDefectsList(QFrame):
    """Vertical, scrollable-when-needed list of recent fault entries."""

    def __init__(self, *, max_entries: int = 20, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._max = max_entries
        self._entries: list[tuple[datetime, str]] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(4)

        self._title = QLabel("Recent Defects")
        self._title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        self._layout.addWidget(self._title)

        self._content = QVBoxLayout()
        self._content.setSpacing(2)
        self._layout.addLayout(self._content)
        self._layout.addStretch(1)

        self._empty = QLabel("No defects yet.")
        self._empty.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._content.addWidget(self._empty)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def add_entry(self, reason: str, *, when: datetime | None = None) -> None:
        when = when or datetime.now()
        self._entries.insert(0, (when, reason))
        if len(self._entries) > self._max:
            self._entries = self._entries[: self._max]
        self._rebuild()

    def clear(self) -> None:
        self._entries.clear()
        self._rebuild()

    def _rebuild(self) -> None:
        # Wipe content layout.
        while self._content.count():
            item = self._content.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        if not self._entries:
            empty = QLabel("No defects yet.")
            empty.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
            self._content.addWidget(empty)
            return
        for when, reason in self._entries:
            row = QLabel(f"{when.strftime('%H:%M:%S')}  ·  {reason}")
            row.setStyleSheet(
                f"color: {ACCENT_RED}; font-size: {FONT_SMALL}pt;"
                f" border-top: 1px solid {BORDER}; padding-top: 2px;"
            )
            row.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self._content.addWidget(row)
