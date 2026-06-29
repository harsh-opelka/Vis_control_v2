"""PLC signal indicator: name + colored dot + TRUE/FALSE."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from viscontrol.ui.theme import ACCENT_RED, BORDER, FONT_NORMAL, SUCCESS_GREEN, TEXT_PRIMARY


class _Dot(QWidget):
    """Small filled circle whose color is set externally."""

    def __init__(self, color: str = SUCCESS_GREEN, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._color = QColor(color)
        self._glow = False

    def set_color(self, color: str, *, glow: bool = False) -> None:
        self._color = QColor(color)
        self._glow = glow
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._glow:
            # Outer halo: same color at lower alpha.
            halo = QColor(self._color)
            halo.setAlpha(80)
            p.setBrush(halo)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(0, 0, 14, 14)
            p.setBrush(self._color)
            p.drawEllipse(2, 2, 10, 10)
        else:
            p.setBrush(self._color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(2, 2, 10, 10)


class SignalPill(QFrame):
    """Row of: signal name | dot | TRUE/FALSE.

    The ``healthy_when`` arg flips the color logic. For ``TuchabzugRunning``
    we want green when the line is running (the "normal" production state);
    for ``FaultActive`` we want green when it's false. ``StopTuchabzug`` is
    informational — both states are colored, with red glow when asserted.
    """

    def __init__(
        self,
        name: str,
        *,
        healthy_when: bool = False,
        glow_when_true: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._healthy_when = healthy_when
        self._glow_when_true = glow_when_true
        self._value = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        self._name = QLabel(name)
        self._name.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-family: 'Consolas','Courier New',monospace;"
            f" font-size: {FONT_NORMAL}pt;"
        )

        self._dot = _Dot()
        self._dot.set_color(BORDER)  # initial neutral

        self._state = QLabel("FALSE")
        self._state.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt;")
        self._state.setFixedWidth(60)
        self._state.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._name)
        layout.addStretch(1)
        layout.addWidget(self._dot)
        layout.addWidget(self._state)

    def set_value(self, value: bool) -> None:
        self._value = bool(value)
        self._state.setText("TRUE" if self._value else "FALSE")
        is_healthy = self._value is self._healthy_when
        color = SUCCESS_GREEN if is_healthy else ACCENT_RED
        glow = self._glow_when_true and self._value
        self._dot.set_color(color, glow=glow)
