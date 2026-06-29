"""Centralised theme constants.

Every widget reads colors and small layout constants from this module. No
hex literal lives anywhere else in the UI code — if you find yourself wanting
to hardcode a color, add it here instead so the SERVICE view can theme it
later without grepping every file.
"""

from __future__ import annotations

from viscontrol.core.events import State

# ---------- palette ----------

NAVY_PRIMARY = "#1B3E7F"
NAVY_DARK = "#15315F"
ACCENT_RED = "#C8102E"
BACKGROUND = "#FAFAF7"
CARD_WHITE = "#FFFFFF"
SUCCESS_GREEN = "#2D8659"
WARNING_AMBER = "#D9941A"
TEXT_PRIMARY = "#2C2C2A"
TEXT_SECONDARY = "#5F5E5A"
BORDER = "#E5E3DC"
TRANSFER_LINE = "#FAC775"


# ---------- state-color mapping ----------

STATE_COLOR = {
    State.WAITING: TEXT_SECONDARY,
    State.TRACKING: NAVY_PRIMARY,
    State.INSPECTING: NAVY_PRIMARY,
    State.READY: SUCCESS_GREEN,
    State.FAULT: ACCENT_RED,
    State.SERVICE: WARNING_AMBER,
}


def state_color(state: State) -> str:
    return STATE_COLOR.get(state, TEXT_SECONDARY)


# ---------- detection overlay colors ----------

OVERLAY_SINGLE = SUCCESS_GREEN
OVERLAY_ROW_FUSED = ACCENT_RED
OVERLAY_COLUMN_FUSED = WARNING_AMBER
OVERLAY_UNKNOWN = ACCENT_RED


# ---------- layout constants ----------

SIDEBAR_WIDTH = 240
CARD_RADIUS = 12
BUTTON_RADIUS = 8
STAT_CARD_WIDTH = 120
STAT_CARD_HEIGHT = 80


# ---------- font sizes (in points) ----------

FONT_HUGE = 64
FONT_LARGE = 28
FONT_BIG_NUMBER = 36
FONT_NORMAL = 12
FONT_SMALL = 10
FONT_MONO = 11


# ---------- stylesheets ----------

GLOBAL_STYLESHEET = f"""
QMainWindow, QWidget#centralWidget {{
    background-color: {BACKGROUND};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", sans-serif;
    font-size: {FONT_NORMAL}pt;
}}

QFrame#sidebar {{
    background-color: {NAVY_PRIMARY};
    color: white;
}}

QFrame#sidebar QLabel {{
    color: white;
}}

QFrame#sidebar QPushButton {{
    background-color: transparent;
    color: white;
    text-align: left;
    padding: 12px 20px;
    border: none;
    font-size: {FONT_NORMAL}pt;
}}

QFrame#sidebar QPushButton:hover {{
    background-color: {NAVY_DARK};
}}

QFrame#sidebar QPushButton[selected="true"] {{
    background-color: {NAVY_DARK};
    border-left: 4px solid {ACCENT_RED};
    padding-left: 16px;
}}

QFrame#card {{
    background-color: {CARD_WHITE};
    border-radius: {CARD_RADIUS}px;
    border: 1px solid {BORDER};
}}

QPushButton#primary {{
    background-color: {NAVY_PRIMARY};
    color: white;
    border-radius: {BUTTON_RADIUS}px;
    padding: 10px 20px;
    font-weight: bold;
}}

QPushButton#primary:hover {{ background-color: {NAVY_DARK}; }}
QPushButton#primary:disabled {{ background-color: {BORDER}; color: {TEXT_SECONDARY}; }}

QPushButton#stop {{
    background-color: {ACCENT_RED};
    color: white;
    border-radius: {BUTTON_RADIUS}px;
    padding: 10px 20px;
    font-weight: bold;
}}

QPushButton#secondary {{
    background-color: {CARD_WHITE};
    color: {NAVY_PRIMARY};
    border: 1px solid {NAVY_PRIMARY};
    border-radius: {BUTTON_RADIUS}px;
    padding: 8px 16px;
}}

QPushButton#secondary:hover {{ background-color: {BORDER}; }}

QStatusBar {{
    background-color: {CARD_WHITE};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER};
    font-size: {FONT_SMALL}pt;
}}

QLabel[role="hint"] {{
    color: {TEXT_SECONDARY};
    font-size: {FONT_SMALL}pt;
}}

QLineEdit, QComboBox, QSpinBox {{
    padding: 6px 10px;
    border: 1px solid {BORDER};
    border-radius: {BUTTON_RADIUS}px;
    background: {CARD_WHITE};
}}

QToolTip {{
    background-color: {NAVY_DARK};
    color: white;
    border: none;
    padding: 4px 6px;
}}

QFrame#startupBanner {{
    background-color: {SUCCESS_GREEN};
    border: none;
    border-radius: 0px;
}}

QFrame#startupBanner QLabel {{
    color: white;
    background: transparent;
    font-size: {FONT_NORMAL}pt;
}}
"""
