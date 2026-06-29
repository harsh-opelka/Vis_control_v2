"""Large state label that changes color by state."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from viscontrol.core.events import State
from viscontrol.ui.theme import FONT_HUGE, FONT_LARGE, state_color


class StateBanner(QLabel):
    """Centered text label tinted by the current state.

    Pass ``compact=True`` in the inference view to use a smaller font and a
    fixed height, giving the camera row more vertical space.
    """

    def __init__(self, compact: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._compact = compact
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if compact:
            self.setFixedHeight(40)
        self.set_state(State.WAITING)

    def set_state(self, state: State) -> None:
        self.set_display(state.value, state_color(state))

    def set_display(self, label: str, color: str) -> None:
        """Set the banner text and color directly, bypassing the State enum."""
        font_size = FONT_LARGE if self._compact else FONT_HUGE
        self.setStyleSheet(
            f"color: {color}; font-size: {font_size}pt; font-weight: 800; letter-spacing: 4px;"
        )
        self.setText(label)
