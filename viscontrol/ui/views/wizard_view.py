"""Installation Wizard — 5 camera-focused steps.

Step 1: Image Orientation  — live preview + rotation/flip controls.
Step 2: ROI Boundaries     — full frame + draggable blue line (roi_split_x).
Step 3: Transfer Line      — cloth ROI + draggable yellow line (transfer_line_x).
Step 4: Exposure & Gain    — live frame + sliders + brightness histogram.
Step 5: Calibration        — cloth ROI with live detection overlays + sensitivity slider.

Settings are committed to the active profile when the user clicks Next on each
step, not deferred to Finish.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from viscontrol.core.config import _DetectionSection
from viscontrol.core.profiles import CropRegion, ProductProfile
from viscontrol.detection.classical import ClassicalDetector as _ClassicalDetector
from viscontrol.detection.pipeline import InspectionPipeline
from viscontrol.ui.theme import (
    FONT_LARGE,
    FONT_NORMAL,
    FONT_SMALL,
    SUCCESS_GREEN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TRANSFER_LINE,
)
from viscontrol.ui.widgets.camera_view import FramePreview


# ─────────────────────────────────────────────────────────────────────────────
# Private canvas widgets
# ─────────────────────────────────────────────────────────────────────────────


class _DraggableLineCanvas(QWidget):
    """Full-frame (or ROI) viewer with a single draggable vertical line.

    Coordinates emitted via ``line_x_changed`` are always in *image space*,
    i.e. raw pixel columns of the numpy array passed to ``set_frame``.
    """

    line_x_changed = Signal(int)

    def __init__(
        self,
        *,
        line_color: str = "#4A9EF7",
        label_prefix: str = "x",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._line_x_img: int = 0
        self._dragging = False
        self._line_color = QColor(line_color)
        self._label_prefix = label_prefix
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._scale: float = 1.0
        self._dx: int = 0
        self._dy: int = 0
        self._img_w: int = 1
        self._img_h: int = 1

    # ── public ───────────────────────────────────────────────────────────────

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._img_h, self._img_w = frame.shape[:2]
        # Re-clamp stored position now that we know the actual image dimensions.
        # This corrects the case where set_line_x() was called before any frame arrived
        # (when _img_w was still the sentinel value 1).
        self._line_x_img = max(0, min(self._line_x_img, self._img_w - 1))
        self._recompute_transform()
        self.update()

    def set_line_x(self, x: int) -> None:
        # Avoid clamping against the sentinel _img_w=1 (no frame yet).
        # Clamp properly once a real frame arrives via set_frame().
        if self._img_w > 1:
            self._line_x_img = max(0, min(x, self._img_w - 1))
        else:
            self._line_x_img = max(0, x)
        self.update()

    # ── coordinate helpers ───────────────────────────────────────────────────

    def _recompute_transform(self) -> None:
        w, h = self.width(), self.height()
        sx = w / max(self._img_w, 1)
        sy = h / max(self._img_h, 1)
        self._scale = min(sx, sy)
        self._dx = (w - int(self._img_w * self._scale)) // 2
        self._dy = (h - int(self._img_h * self._scale)) // 2

    def _to_wx(self, img_x: int) -> int:
        return self._dx + int(img_x * self._scale)

    def _to_ix(self, wx: float) -> int:
        return max(0, min(self._img_w - 1, int((wx - self._dx) / max(self._scale, 1e-9))))

    def _near_line(self, wx: float) -> bool:
        return abs(wx - self._to_wx(self._line_x_img)) <= 8

    # ── Qt events ────────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:  # noqa: D401
        self._recompute_transform()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: D401
        if event.button() == Qt.MouseButton.LeftButton and self._near_line(
            event.position().x()
        ):
            self._dragging = True

    def mouseMoveEvent(self, event) -> None:  # noqa: D401
        wx = event.position().x()
        if self._dragging:
            ix = self._to_ix(wx)
            if ix != self._line_x_img:
                self._line_x_img = ix
                self.line_x_changed.emit(ix)
                self.update()
        self.setCursor(
            Qt.CursorShape.SizeHorCursor
            if self._near_line(wx)
            else Qt.CursorShape.ArrowCursor
        )

    def mouseReleaseEvent(self, event) -> None:  # noqa: D401
        self._dragging = False

    def paintEvent(self, event) -> None:  # noqa: D401
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#1A1A1A"))

        if self._frame is None:
            p.setPen(QColor(TEXT_SECONDARY))
            font = QFont()
            font.setPointSize(FONT_NORMAL)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no signal")
            return

        img = self._frame
        h, w = img.shape[:2]
        pw, ph = int(w * self._scale), int(h * self._scale)
        if img.ndim == 2:
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img.tobytes(), w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            pw, ph,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        p.drawPixmap(self._dx, self._dy, pixmap)

        lx = self._to_wx(self._line_x_img)
        bot = self._dy + ph
        pen = QPen(self._line_color)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(lx, self._dy, lx, bot)

        # Circular drag handles at top and bottom edges of the frame
        p.setBrush(self._line_color)
        p.setPen(Qt.PenStyle.NoPen)
        for ty in (self._dy + 7, bot - 7):
            p.drawEllipse(lx - 5, ty - 5, 10, 10)

        font = QFont()
        font.setPointSize(FONT_SMALL)
        font.setBold(True)
        p.setFont(font)
        p.setPen(self._line_color)
        p.drawText(lx + 9, self._dy + 22, f"{self._label_prefix} = {self._line_x_img} px")


class _HistogramWidget(QWidget):
    """64-bin brightness histogram rendered with QPainter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(220, 70)
        self._hist: np.ndarray | None = None
        self.setStyleSheet("background-color: #1A1A1A; border-radius: 4px;")

    def set_frame(self, frame: np.ndarray) -> None:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        mx = hist.max()
        self._hist = hist / mx if mx > 0 else hist
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1A1A1A"))
        if self._hist is None:
            return
        w, h = self.width(), self.height()
        pad = 4
        n = len(self._hist)
        bar_w = (w - 2 * pad) / n
        p.setPen(Qt.PenStyle.NoPen)
        for i, v in enumerate(self._hist):
            bar_h = int(v * (h - 2 * pad))
            shade = int(80 + (i / (n - 1)) * 175)
            p.setBrush(QColor(shade, shade, shade))
            p.drawRect(
                int(pad + i * bar_w), h - pad - bar_h,
                max(1, int(bar_w) - 1), bar_h,
            )
        # Under/over-exposure guide lines (5 % and 95 %)
        pen = QPen(QColor("#D9941A"))
        pen.setStyle(Qt.PenStyle.DotLine)
        p.setPen(pen)
        for frac in (0.05, 0.95):
            x = int(pad + frac * (w - 2 * pad))
            p.drawLine(x, pad, x, h - pad)


class _CalibHistogramWidget(QWidget):
    """400 × 100 brightness histogram with a red threshold line for the calibration step."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(400, 100)
        self._hist: np.ndarray | None = None
        self._threshold: int | None = None
        self.setStyleSheet("background-color: #1A1A1A; border-radius: 4px;")

    def set_data(self, frame: np.ndarray, threshold: int | None) -> None:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        mx = hist.max()
        self._hist = hist / mx if mx > 0 else hist
        self._threshold = threshold
        self.update()

    def paintEvent(self, event) -> None:  # noqa: D401
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1A1A1A"))
        if self._hist is None:
            p.setPen(QColor(TEXT_SECONDARY))
            font = QFont()
            font.setPointSize(FONT_SMALL)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no signal")
            return
        w, h = self.width(), self.height()
        pad = 4
        n = len(self._hist)
        bar_w = (w - 2 * pad) / n
        p.setPen(Qt.PenStyle.NoPen)
        for i, v in enumerate(self._hist):
            bar_h = int(v * (h - 2 * pad))
            shade = int(60 + (i / max(n - 1, 1)) * 195)
            p.setBrush(QColor(shade, shade, shade))
            p.drawRect(
                int(pad + i * bar_w), h - pad - bar_h,
                max(1, int(bar_w)), bar_h,
            )
        if self._threshold is not None:
            tx = int(pad + self._threshold * (w - 2 * pad) / 255)
            pen = QPen(QColor("#FF4444"))
            pen.setWidth(2)
            p.setPen(pen)
            p.drawLine(tx, pad, tx, h - pad)


class _DetectionCanvas(QWidget):
    """Cloth-ROI viewer with numbered detection overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._detections: list = []
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._scale: float = 1.0
        self._dx = 0
        self._dy = 0
        self._img_w = 1
        self._img_h = 1

    def set_data(self, frame: np.ndarray, detections: list) -> None:
        self._frame = frame
        self._detections = detections
        self._img_h, self._img_w = frame.shape[:2]
        w, h = self.width(), self.height()
        sx = w / max(self._img_w, 1)
        sy = h / max(self._img_h, 1)
        self._scale = min(sx, sy)
        self._dx = (w - int(self._img_w * self._scale)) // 2
        self._dy = (h - int(self._img_h * self._scale)) // 2
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: D401
        if self._frame is not None:
            self.set_data(self._frame, self._detections)
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: D401
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#1A1A1A"))

        if self._frame is None:
            p.setPen(QColor(TEXT_SECONDARY))
            font = QFont()
            font.setPointSize(FONT_NORMAL)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no signal")
            return

        img = self._frame
        h, w = img.shape[:2]
        pw, ph = int(w * self._scale), int(h * self._scale)
        if img.ndim == 2:
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img.tobytes(), w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            pw, ph,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        p.drawPixmap(self._dx, self._dy, pixmap)

        sx = self._scale
        font = QFont()
        font.setPointSize(FONT_SMALL)
        font.setBold(True)
        p.setFont(font)

        single_n = 1
        for det in self._detections:
            cx, cy = det.centroid
            wx = self._dx + cx * sx
            wy = self._dy + cy * sx
            r = max(det.width_px, det.height_px) / 2 * sx

            if det.label == "single":
                pen = QPen(QColor(SUCCESS_GREEN))
                pen.setWidth(2)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(wx - r), int(wy - r), int(r * 2), int(r * 2))
                p.setPen(QColor(SUCCESS_GREEN))
                p.drawText(int(wx + r + 4), int(wy + 5), str(single_n))
                single_n += 1
            else:
                pen = QPen(QColor("#D9941A"))
                pen.setWidth(2)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(wx - r), int(wy - r), int(r * 2), int(r * 2))


class _CropCanvas(QWidget):
    """ROI viewer with four draggable crop lines (top/bottom/left/right).

    The crop values are measured in pixels from each edge of the *image*
    (i.e. the same as ``CropRegion.top/bottom/left/right``).  Dragging the
    top line inward increases ``top``; dragging the bottom line inward
    increases ``bottom``; and so on.

    ``crop_changed(top, bottom, left, right)`` is emitted on every drag tick.
    """

    crop_changed = Signal(int, int, int, int)

    _LINE_COLOR = "#00C8FF"
    _SNAP = 8

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        self._top = 0
        self._bottom = 0
        self._left = 0
        self._right = 0
        self._img_h = 1
        self._img_w = 1
        self._scale = 1.0
        self._dx = 0
        self._dy = 0
        self._dragging: str | None = None  # "top" | "bottom" | "left" | "right"
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # ── public ────────────────────────────────────────────────────────────────

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._img_h, self._img_w = frame.shape[:2]
        self._recompute()
        self.update()

    def set_crop(self, top: int, bottom: int, left: int, right: int) -> None:
        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right
        self.update()

    # ── transform helpers ────────────────────────────────────────────────────

    def _recompute(self) -> None:
        w, h = self.width(), self.height()
        self._scale = min(w / max(self._img_w, 1), h / max(self._img_h, 1))
        self._dx = (w - int(self._img_w * self._scale)) // 2
        self._dy = (h - int(self._img_h * self._scale)) // 2

    def _wx(self, ix: int) -> int:
        return self._dx + int(ix * self._scale)

    def _wy(self, iy: int) -> int:
        return self._dy + int(iy * self._scale)

    def _to_ix(self, wx: float) -> int:
        return max(0, min(self._img_w, int((wx - self._dx) / max(self._scale, 1e-9))))

    def _to_iy(self, wy: float) -> int:
        return max(0, min(self._img_h, int((wy - self._dy) / max(self._scale, 1e-9))))

    # Widget positions of the 4 lines
    def _wy_top(self) -> int: return self._wy(self._top)
    def _wy_bot(self) -> int: return self._wy(self._img_h - self._bottom)
    def _wx_left(self) -> int: return self._wx(self._left)
    def _wx_right(self) -> int: return self._wx(self._img_w - self._right)

    def _hit(self, wx: float, wy: float) -> str | None:
        if abs(wy - self._wy_top()) <= self._SNAP:
            return "top"
        if abs(wy - self._wy_bot()) <= self._SNAP:
            return "bottom"
        if abs(wx - self._wx_left()) <= self._SNAP:
            return "left"
        if abs(wx - self._wx_right()) <= self._SNAP:
            return "right"
        return None

    # ── Qt events ────────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:  # noqa: D401
        if self._frame is not None:
            self._recompute()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: D401
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = self._hit(event.position().x(), event.position().y())

    def mouseMoveEvent(self, event) -> None:  # noqa: D401
        wx, wy = event.position().x(), event.position().y()
        if self._dragging:
            if self._dragging == "top":
                iy = self._to_iy(wy)
                self._top = max(0, min(iy, self._img_h - 1 - self._bottom))
            elif self._dragging == "bottom":
                iy = self._to_iy(wy)
                new_b = max(0, self._img_h - iy)
                self._bottom = min(new_b, self._img_h - 1 - self._top)
            elif self._dragging == "left":
                ix = self._to_ix(wx)
                self._left = max(0, min(ix, self._img_w - 1 - self._right))
            elif self._dragging == "right":
                ix = self._to_ix(wx)
                new_r = max(0, self._img_w - ix)
                self._right = min(new_r, self._img_w - 1 - self._left)
            self.crop_changed.emit(self._top, self._bottom, self._left, self._right)
            self.update()
        hit = self._hit(wx, wy)
        if hit in ("top", "bottom"):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif hit in ("left", "right"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: D401
        self._dragging = None

    def paintEvent(self, event) -> None:  # noqa: D401
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#1A1A1A"))

        if self._frame is None:
            p.setPen(QColor(TEXT_SECONDARY))
            font = QFont()
            font.setPointSize(FONT_NORMAL)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no signal")
            return

        img = self._frame
        h, w = img.shape[:2]
        pw, ph = int(w * self._scale), int(h * self._scale)
        if img.ndim == 2:
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img.tobytes(), w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            pw, ph, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        p.drawPixmap(self._dx, self._dy, pixmap)

        wy_t = self._wy_top()
        wy_b = self._wy_bot()
        wx_l = self._wx_left()
        wx_r = self._wx_right()
        x0, y0 = self._dx, self._dy

        # Dim outside-crop areas
        dim = QColor(0, 0, 0, 120)
        if wy_t > y0:
            p.fillRect(x0, y0, pw, wy_t - y0, dim)
        if wy_b < y0 + ph:
            p.fillRect(x0, wy_b, pw, y0 + ph - wy_b, dim)
        if wx_l > x0:
            p.fillRect(x0, wy_t, wx_l - x0, wy_b - wy_t, dim)
        if wx_r < x0 + pw:
            p.fillRect(wx_r, wy_t, x0 + pw - wx_r, wy_b - wy_t, dim)

        # Crop lines
        color = QColor(self._LINE_COLOR)
        pen = QPen(color)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(x0, wy_t, x0 + pw, wy_t)
        p.drawLine(x0, wy_b, x0 + pw, wy_b)
        p.drawLine(wx_l, y0, wx_l, y0 + ph)
        p.drawLine(wx_r, y0, wx_r, y0 + ph)

        # Drag handles at midpoints of each line
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        mid_x = (wx_l + wx_r) // 2
        mid_y = (wy_t + wy_b) // 2
        for hx, hy in [(mid_x, wy_t), (mid_x, wy_b), (wx_l, mid_y), (wx_r, mid_y)]:
            p.drawEllipse(hx - 5, hy - 5, 10, 10)

        # Labels
        font = QFont()
        font.setPointSize(FONT_SMALL)
        font.setBold(True)
        p.setFont(font)
        p.setPen(color)
        p.drawText(x0 + 8, wy_t + 16, f"top={self._top}px")
        p.drawText(x0 + 8, wy_b - 6, f"bottom={self._bottom}px")
        p.drawText(wx_l + 4, y0 + 16, f"left={self._left}px")
        right_lbl = f"right={self._right}px"
        p.drawText(wx_r - len(right_lbl) * 6, y0 + 16, right_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Main wizard widget
# ─────────────────────────────────────────────────────────────────────────────


class WizardView(QWidget):
    """7-step installation wizard (camera-focused)."""

    completed = Signal(dict)
    cancelled = Signal()
    apply_orientation_requested = Signal(int, bool)   # rotation, flip_horizontal
    learn_reference_requested = Signal(str)            # profile name
    roi_split_x_changed = Signal(int)
    transfer_line_x_changed = Signal(int)
    belt_crop_changed = Signal(int, int, int, int)     # top, bottom, left, right
    cloth_crop_changed = Signal(int, int, int, int)
    belt_dough_is_darker_changed = Signal(bool)
    exposure_changed = Signal(int)                     # μs, committed on slider release
    gain_changed = Signal(float)                       # dB, committed on slider release
    noise_threshold_changed = Signal(float)            # committed on sensitivity release
    dough_is_darker_changed = Signal(bool)             # committed immediately on toggle
    # FIX 2: cloth detection method, changeable from the Calibration page —
    # same shared setting as the Service page's combo (see MainWindow wiring).
    detection_method_changed = Signal(str)
    # Row grouping (per-row stop): emits the new on/off state; MainWindow
    # persists it to config.detection.use_row_grouping (shared app-level flag).
    row_grouping_changed = Signal(bool)
    # Cloth Reference (Hough gating): emits the chosen brightness threshold;
    # MainWindow._on_save_cloth_reference captures the empty-cloth mask at that
    # threshold and persists it (config.detection.hough.cloth_reference_path).
    save_cloth_reference_requested = Signal(int)
    # SECTION 4: transfer bridge width (px, profile field). MainWindow persists
    # it to the active profile's transfer_bridge_width_px.
    transfer_bridge_width_changed = Signal(int)
    # SECTION 6: grid columns per row and number of rows (config.detection.*).
    grid_columns_changed = Signal(int)
    grid_rows_changed = Signal(int)

    STEPS = [
        "Orientation",
        "ROI Boundaries",
        "Transfer Line",
        "Belt area",
        "Cloth area",
        "Exposure & Gain",
        "Calibration",
    ]

    _DEFAULT_ROI_SPLIT_X = 1768
    _DEFAULT_TRANSFER_LINE_X = 2400
    _DEFAULT_EXPOSURE_US = 2000
    _DEFAULT_GAIN = 0.0
    _DEFAULT_BRIDGE_WIDTH_PX = 40   # SECTION 4
    _DEFAULT_GRID_COLUMNS = 8       # SECTION 6
    _DEFAULT_GRID_ROWS = 2          # default number of rows on the cloth

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_step = 0
        self._current_roi_split_x: int = self._DEFAULT_ROI_SPLIT_X
        self._current_transfer_line_x: int = self._DEFAULT_TRANSFER_LINE_X
        self._current_exposure_us: int = self._DEFAULT_EXPOSURE_US
        self._current_gain: float = self._DEFAULT_GAIN
        self._current_profile: ProductProfile | None = None
        self._calibration_detector = _ClassicalDetector()
        # Crop state: (top, bottom, left, right) in ROI-local pixels
        self._current_belt_crop: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._current_cloth_crop: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._current_belt_dough_is_darker: bool = False
        # FIX 2: mirrors AppConfig.detection (see core/config.py) — pushed in by
        # MainWindow._open_wizard / whenever the method changes from either the
        # Service page or this page, via set_detection_settings(). Selects which
        # of the four cloth detection methods _run_detection_preview dispatches to.
        self._detection_cfg: "_DetectionSection" = _DetectionSection()
        # bg_subtract reference (full cloth ROI, pre-crop) — mirrors
        # MainWindow._bg_reference_cloth; pushed in via set_bg_reference().
        self._bg_reference_cloth: np.ndarray | None = None
        # Cloth Reference mask (Hough gating) — mirrors MainWindow._cloth_reference_mask
        # for the saved-status indicator only; pushed in via set_cloth_reference().
        self._cloth_reference_mask: np.ndarray | None = None

        # Throttle calibration preview: store latest cloth and belt frames and
        # fire at most once per 500 ms (2 fps).  Detection can be slow on
        # high-res images; running it on every camera frame blocks the GUI thread.
        self._pending_cloth_frame: np.ndarray | None = None
        self._pending_belt_frame: np.ndarray | None = None
        self._det_throttle_timer = QTimer(self)
        self._det_throttle_timer.setSingleShot(True)
        self._det_throttle_timer.setInterval(500)
        self._det_throttle_timer.timeout.connect(self._dispatch_detection)

        self._build()

    # ── builder ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        header = QFrame()
        header.setObjectName("card")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(20, 12, 20, 12)
        self._title = QLabel(self.tr("Installation Wizard"))
        self._title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_LARGE}pt; font-weight: 700;"
        )
        self._subtitle = QLabel("")
        self._subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hl.addWidget(self._title)
        hl.addWidget(self._subtitle)
        root.addWidget(header, 0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_step1_orientation())
        self._stack.addWidget(self._build_step2_roi())
        self._stack.addWidget(self._build_step3_transfer())
        self._stack.addWidget(self._build_step4_belt_crop())
        self._stack.addWidget(self._build_step5_cloth_crop())
        self._stack.addWidget(self._build_step6_exposure())
        self._stack.addWidget(self._build_step7_calibration())
        root.addWidget(self._stack, 1)

        nav = QHBoxLayout()
        self._cancel_btn = QPushButton(self.tr("Cancel"))
        self._cancel_btn.setObjectName("secondary")
        self._cancel_btn.clicked.connect(self.cancelled.emit)
        self._back_btn = QPushButton(self.tr("Back"))
        self._back_btn.setObjectName("secondary")
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn = QPushButton(self.tr("Next"))
        self._next_btn.setObjectName("primary")
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._cancel_btn)
        nav.addStretch(1)
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)
        self._update_nav()

    # ── step 1: orientation ──────────────────────────────────────────────────

    def _build_step1_orientation(self) -> QWidget:
        page = QFrame()
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(16)

        self._wiz_orient_preview = FramePreview(width=480, height=360)
        layout.addWidget(self._wiz_orient_preview, 0, Qt.AlignmentFlag.AlignHCenter)

        controls = QHBoxLayout()
        controls.addStretch(1)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._wiz_rotation = QComboBox()
        for v in (0, 90, 180, 270):
            self._wiz_rotation.addItem(f"{v}°", v)
        self._wiz_flip = QCheckBox(self.tr("Flip horizontally"))
        form.addRow(self.tr("Rotation"), self._wiz_rotation)
        form.addRow("", self._wiz_flip)
        controls.addLayout(form)
        controls.addStretch(1)
        layout.addLayout(controls)

        def _emit_apply() -> None:
            self.apply_orientation_requested.emit(
                self._wiz_rotation.currentData(),
                self._wiz_flip.isChecked(),
            )

        self._wiz_rotation.currentIndexChanged.connect(lambda _: _emit_apply())
        self._wiz_flip.toggled.connect(lambda _: _emit_apply())
        return page

    # ── step 2: ROI boundaries ───────────────────────────────────────────────

    def _build_step2_roi(self) -> QWidget:
        page = QFrame()
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel(
            self.tr(
                "Drag the blue line to set the boundary between the "
                "belt (left) and the cloth (right)."
            )
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._roi_canvas = _DraggableLineCanvas(
            line_color="#4A9EF7", label_prefix="roi_split_x"
        )
        self._roi_canvas.set_line_x(self._current_roi_split_x)
        self._roi_canvas.line_x_changed.connect(self._on_roi_dragged)
        layout.addWidget(self._roi_canvas, 1)

        self._roi_x_lbl = QLabel(f"roi_split_x = {self._current_roi_split_x} px")
        self._roi_x_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;")
        layout.addWidget(self._roi_x_lbl)
        return page

    def _on_roi_dragged(self, x: int) -> None:
        """x is a FULL-FRAME column.

        _roi_canvas receives the full camera frame so _DraggableLineCanvas._to_ix
        converts the widget drag position directly into full-frame image coordinates.
        """
        self._current_roi_split_x = x
        self._roi_x_lbl.setText(f"roi_split_x = {x} px")

    # ── step 3: transfer line ────────────────────────────────────────────────

    def _build_step3_transfer(self) -> QWidget:
        page = QFrame()
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel(
            self.tr(
                "Drag the yellow line to set the transfer-line position on the cloth. "
                "When dough reaches this line StopTuchabzug is triggered."
            )
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._tl_canvas = _DraggableLineCanvas(
            line_color=TRANSFER_LINE, label_prefix="transfer_line"
        )
        tl_local = max(0, self._current_transfer_line_x - self._current_roi_split_x)
        self._tl_canvas.set_line_x(tl_local)
        self._tl_canvas.line_x_changed.connect(self._on_tl_dragged)
        layout.addWidget(self._tl_canvas, 1)

        self._tl_x_lbl = QLabel(
            f"transfer_line_x = {self._current_transfer_line_x} px (full frame)"
        )
        self._tl_x_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;")
        layout.addWidget(self._tl_x_lbl)

        # SECTION 4: transfer-bridge width. The bridge is a band centered on the
        # transfer line; a piece counts as "at the transfer point" once its
        # leading edge overlaps it. Widen it for faster conveyors / more frame
        # skipping so a row can't slip across between processed frames.
        bridge_row = QHBoxLayout()
        bridge_lbl = QLabel(self.tr("Transfer bridge width (px):"))
        bridge_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._bridge_width_spin = QSpinBox()
        self._bridge_width_spin.setRange(1, 2000)
        self._bridge_width_spin.setSingleStep(5)
        self._bridge_width_spin.setValue(self._DEFAULT_BRIDGE_WIDTH_PX)
        self._bridge_width_spin.valueChanged.connect(self.transfer_bridge_width_changed.emit)
        bridge_hint = QLabel(
            self.tr("Wider = more robust to dropped frames / fast conveyor.")
        )
        bridge_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        bridge_row.addWidget(bridge_lbl, 0)
        bridge_row.addWidget(self._bridge_width_spin, 0)
        bridge_row.addWidget(bridge_hint, 1)
        layout.addLayout(bridge_row)
        return page

    def _on_tl_dragged(self, local_x: int) -> None:
        """local_x is a CLOTH-LOCAL column.

        _tl_canvas receives the cloth crop (frame[:, roi_split_x:]), so
        _DraggableLineCanvas._to_ix returns a position within that crop.
        Add roi_split_x here to reconstitute the full-frame column before saving.
        """
        self._current_transfer_line_x = self._current_roi_split_x + local_x
        self._tl_x_lbl.setText(
            f"transfer_line_x = {self._current_transfer_line_x} px (full frame)"
        )

    # ── step 4: belt crop area ────────────────────────────────────────────────

    def _build_step4_belt_crop(self) -> QWidget:
        page = QFrame()
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel(
            self.tr(
                "Drag the lines to crop the belt view to only the area where "
                "dough lands. Dark machine-frame regions outside the crop are "
                "ignored by detection."
            )
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._belt_crop_canvas = _CropCanvas()
        self._belt_crop_canvas.set_crop(*self._current_belt_crop)
        self._belt_crop_canvas.crop_changed.connect(self._on_belt_crop_changed)
        layout.addWidget(self._belt_crop_canvas, 1)

        self._belt_crop_lbl = QLabel(self._crop_text("belt", self._current_belt_crop))
        self._belt_crop_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;")
        layout.addWidget(self._belt_crop_lbl)

        # Belt detection polarity — usually opposite of cloth:
        # belt is a dark wire-mesh, dough is lighter → default unchecked (not darker).
        self._belt_darker_check = QCheckBox(
            self.tr("Belt: dough darker than belt background")
        )
        self._belt_darker_check.setChecked(self._current_belt_dough_is_darker)
        self._belt_darker_check.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;"
        )
        belt_polar_hint = QLabel(
            self.tr(
                "Leave OFF for a dark wire-mesh belt (dough appears lighter). "
                "Enable only if your belt surface is brighter than the dough."
            )
        )
        belt_polar_hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        belt_polar_hint.setWordWrap(True)
        layout.addWidget(self._belt_darker_check)
        layout.addWidget(belt_polar_hint)

        def _on_belt_darker(checked: bool) -> None:
            self._current_belt_dough_is_darker = checked
            self.belt_dough_is_darker_changed.emit(checked)

        self._belt_darker_check.toggled.connect(_on_belt_darker)
        return page

    def _on_belt_crop_changed(self, top: int, bottom: int, left: int, right: int) -> None:
        self._current_belt_crop = (top, bottom, left, right)
        self._belt_crop_lbl.setText(self._crop_text("belt", self._current_belt_crop))

    # ── step 5: cloth crop area ───────────────────────────────────────────────

    def _build_step5_cloth_crop(self) -> QWidget:
        page = QFrame()
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel(
            self.tr(
                "Drag the lines to crop the cloth view to only the area where "
                "dough sits on the Gärtuch. Dark borders are ignored."
            )
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._cloth_crop_canvas = _CropCanvas()
        self._cloth_crop_canvas.set_crop(*self._current_cloth_crop)
        self._cloth_crop_canvas.crop_changed.connect(self._on_cloth_crop_changed)
        layout.addWidget(self._cloth_crop_canvas, 1)

        self._cloth_crop_lbl = QLabel(self._crop_text("cloth", self._current_cloth_crop))
        self._cloth_crop_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;")
        layout.addWidget(self._cloth_crop_lbl)
        return page

    def _on_cloth_crop_changed(self, top: int, bottom: int, left: int, right: int) -> None:
        self._current_cloth_crop = (top, bottom, left, right)
        self._cloth_crop_lbl.setText(self._crop_text("cloth", self._current_cloth_crop))

    @staticmethod
    def _crop_text(roi: str, crop: tuple[int, int, int, int]) -> str:
        t, b, l, r = crop
        return f"{roi}_crop: top={t} bottom={b} left={l} right={r} px"

    # ── step 6: exposure & gain ──────────────────────────────────────────────

    @staticmethod
    def _slider_to_exposure(val: int) -> int:
        """Map slider 0-100 to 10-100 000 μs (log scale)."""
        return max(10, int(10 ** (1.0 + val * 4.0 / 100.0)))

    @staticmethod
    def _exposure_to_slider(us: int) -> int:
        return max(0, min(100, int((math.log10(max(us, 10)) - 1.0) * 25.0)))

    def _build_step6_exposure(self) -> QWidget:
        page = QFrame()
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 16)
        layout.setSpacing(8)

        hint = QLabel(
            self.tr(
                "Adjust exposure and gain so dough pieces are clearly visible "
                "without clipping the highlights."
            )
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._exp_preview = FramePreview(width=600, height=380)
        layout.addWidget(self._exp_preview, 0, Qt.AlignmentFlag.AlignHCenter)

        hist_row = QHBoxLayout()
        hist_row.addStretch(1)
        self._histogram = _HistogramWidget()
        hist_row.addWidget(self._histogram)
        under_over = QLabel(self.tr("← under    over →"))
        under_over.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hist_row.addWidget(under_over)
        hist_row.addStretch(1)
        layout.addLayout(hist_row)

        slider_card = QFrame()
        slider_card.setObjectName("card")
        sc = QFormLayout(slider_card)
        sc.setContentsMargins(20, 12, 20, 12)
        sc.setHorizontalSpacing(16)
        sc.setVerticalSpacing(8)

        self._exp_slider = QSlider(Qt.Orientation.Horizontal)
        self._exp_slider.setRange(0, 100)
        self._exp_slider.setValue(self._exposure_to_slider(self._current_exposure_us))
        self._exp_val_lbl = QLabel(f"{self._current_exposure_us} μs")
        self._exp_val_lbl.setFixedWidth(90)
        exp_row = QHBoxLayout()
        exp_row.addWidget(self._exp_slider, 1)
        exp_row.addWidget(self._exp_val_lbl)
        exp_w = QWidget()
        exp_w.setLayout(exp_row)
        sc.addRow(self.tr("Exposure"), exp_w)

        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, 480)
        self._gain_slider.setValue(int(self._current_gain * 10))
        self._gain_val_lbl = QLabel(f"{self._current_gain:.1f} dB")
        self._gain_val_lbl.setFixedWidth(90)
        gain_row = QHBoxLayout()
        gain_row.addWidget(self._gain_slider, 1)
        gain_row.addWidget(self._gain_val_lbl)
        gain_w = QWidget()
        gain_w.setLayout(gain_row)
        sc.addRow(self.tr("Gain"), gain_w)

        layout.addWidget(slider_card)

        # Display updates on every tick; camera/save triggered on release only.
        def _on_exp_move(val: int) -> None:
            us = self._slider_to_exposure(val)
            self._current_exposure_us = us
            self._exp_val_lbl.setText(f"{us} μs")

        def _on_exp_release() -> None:
            self.exposure_changed.emit(self._current_exposure_us)

        def _on_gain_move(val: int) -> None:
            g = val / 10.0
            self._current_gain = g
            self._gain_val_lbl.setText(f"{g:.1f} dB")

        def _on_gain_release() -> None:
            self.gain_changed.emit(self._current_gain)

        self._exp_slider.valueChanged.connect(_on_exp_move)
        self._exp_slider.sliderReleased.connect(_on_exp_release)
        self._gain_slider.valueChanged.connect(_on_gain_move)
        self._gain_slider.sliderReleased.connect(_on_gain_release)
        return page

    # ── step 7: calibration ──────────────────────────────────────────────────

    def _build_step7_calibration(self) -> QWidget:
        page = QFrame()
        page.setObjectName("card")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        self._calib_count_lbl = QLabel(self.tr("Detecting: — pieces"))
        self._calib_count_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        top_row.addWidget(self._calib_count_lbl)
        top_row.addStretch(1)
        self._learn_btn = QPushButton(self.tr("Learn Reference"))
        self._learn_btn.setObjectName("primary")
        self._learn_btn.setEnabled(False)
        self._learn_btn.setToolTip(self.tr("Need at least 3 detected pieces."))
        self._learn_btn.clicked.connect(
            lambda: self.learn_reference_requested.emit(
                self._current_profile.name if self._current_profile else "Default"
            )
        )
        top_row.addWidget(self._learn_btn)
        layout.addLayout(top_row)

        # FIX 2: cloth detection method — same shared setting as the Service
        # page (Detection section). Switching here re-runs the preview
        # immediately so blob/contour_external/hough/bg_subtract can be
        # A/B compared live without leaving the calibration page.
        method_row = QHBoxLayout()
        method_label = QLabel(self.tr("Cloth detection method:"))
        method_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._calib_method_combo = QComboBox()
        self._calib_method_combo.addItem(self.tr("Blob (default)"), "blob")
        self._calib_method_combo.addItem(self.tr("Contour (external, no solidity)"), "contour_external")
        self._calib_method_combo.addItem(self.tr("Hough circles"), "hough")
        self._calib_method_combo.addItem(self.tr("Background subtraction"), "bg_subtract")
        self._calib_method_combo.currentIndexChanged.connect(self._on_calib_method_changed)
        self._calib_active_method_lbl = QLabel("")
        self._calib_active_method_lbl.setStyleSheet(
            f"color: {SUCCESS_GREEN}; font-size: {FONT_SMALL}pt; font-weight: 600;"
        )
        method_row.addWidget(method_label, 0)
        method_row.addWidget(self._calib_method_combo, 0)
        method_row.addWidget(self._calib_active_method_lbl, 0)
        method_row.addStretch(1)

        # Row grouping (per-row stop) — shared app-level flag
        # (config.detection.use_row_grouping). Off = legacy single-line tripwire
        # (unchanged); on = fire StopTuchabzug once per detected row.
        self._calib_row_grouping_chk = QCheckBox(self.tr("Use row grouping (per-row stop)"))
        self._calib_row_grouping_chk.toggled.connect(self._on_calib_row_grouping_changed)
        self._calib_row_grouping_lbl = QLabel("")
        self._calib_row_grouping_lbl.setStyleSheet(
            f"color: {SUCCESS_GREEN}; font-size: {FONT_SMALL}pt; font-weight: 600;"
        )
        method_row.addWidget(self._calib_row_grouping_chk, 0)
        method_row.addWidget(self._calib_row_grouping_lbl, 0)

        # Grid structure: pieces per row (columns) and total number of rows on
        # the cloth. Both feed the grid-aware tangent stop that identifies Row 1
        # and fires StopTuchabzug when Row 1's trailing tangent hits the bridge.
        cols_lbl = QLabel(self.tr("Columns per row:"))
        cols_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._grid_columns_spin = QSpinBox()
        self._grid_columns_spin.setRange(1, 64)
        self._grid_columns_spin.setValue(self._DEFAULT_GRID_COLUMNS)
        self._grid_columns_spin.valueChanged.connect(self.grid_columns_changed.emit)
        rows_lbl = QLabel(self.tr("Number of rows:"))
        rows_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        self._grid_rows_spin = QSpinBox()
        self._grid_rows_spin.setRange(0, 32)
        self._grid_rows_spin.setSpecialValueText(self.tr("?"))
        self._grid_rows_spin.setValue(self._DEFAULT_GRID_ROWS)
        self._grid_rows_spin.valueChanged.connect(self.grid_rows_changed.emit)
        method_row.addWidget(cols_lbl, 0)
        method_row.addWidget(self._grid_columns_spin, 0)
        method_row.addWidget(rows_lbl, 0)
        method_row.addWidget(self._grid_rows_spin, 0)
        layout.addLayout(method_row)

        self._det_canvas = _DetectionCanvas()
        layout.addWidget(self._det_canvas, 1)

        hint = QLabel(
            self.tr(
                "Not seeing all pieces? Raise the Sensitivity slider. "
                "Check lighting and exposure if the image is too dark."
            )
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Diagnostic panel: brightness histogram + binary mask thumbnail.
        diag_frame = QFrame()
        diag_frame.setObjectName("card")
        diag_layout = QHBoxLayout(diag_frame)
        diag_layout.setContentsMargins(12, 8, 12, 8)
        diag_layout.setSpacing(16)

        hist_col = QVBoxLayout()
        self._calib_thresh_lbl = QLabel(self.tr("Threshold: —"))
        self._calib_thresh_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;"
        )
        hist_col.addWidget(self._calib_thresh_lbl)
        self._calib_hist_widget = _CalibHistogramWidget()
        hist_col.addWidget(self._calib_hist_widget)
        diag_layout.addLayout(hist_col, 1)

        mask_col = QVBoxLayout()
        mask_title = QLabel(self.tr("Cloth Binary Mask"))
        mask_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;")
        mask_col.addWidget(mask_title, 0, Qt.AlignmentFlag.AlignHCenter)
        self._calib_mask_lbl = QLabel()
        self._calib_mask_lbl.setFixedSize(200, 200)
        self._calib_mask_lbl.setStyleSheet("background-color: #000000;")
        self._calib_mask_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_col.addWidget(self._calib_mask_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
        diag_layout.addLayout(mask_col)

        belt_mask_col = QVBoxLayout()
        belt_mask_title = QLabel(self.tr("Belt Binary Mask"))
        belt_mask_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;")
        belt_mask_col.addWidget(belt_mask_title, 0, Qt.AlignmentFlag.AlignHCenter)
        self._calib_belt_mask_lbl = QLabel()
        self._calib_belt_mask_lbl.setFixedSize(200, 200)
        self._calib_belt_mask_lbl.setStyleSheet("background-color: #000000;")
        self._calib_belt_mask_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        belt_mask_col.addWidget(self._calib_belt_mask_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
        diag_layout.addLayout(belt_mask_col)

        layout.addWidget(diag_frame)

        # Cloth Reference (Hough gating, see ClassicalDetector.detect_hough /
        # compute_cloth_region_mask). The cloth/camera don't move during
        # operation, so the bright-cloth-vs-dark-metal split is captured ONCE
        # here on the empty cloth and reused as-is. Tune the threshold against
        # the "Cloth Binary Mask" preview above (live when method=hough), then
        # Save. Only meaningful for the Hough method; harmless otherwise.
        cloth_ref_frame = QFrame()
        cloth_ref_frame.setObjectName("card")
        cloth_ref_layout = QVBoxLayout(cloth_ref_frame)
        cloth_ref_layout.setContentsMargins(12, 8, 12, 8)
        cloth_ref_layout.setSpacing(6)
        cloth_ref_title = QLabel(self.tr("Cloth Reference (Hough gating)"))
        cloth_ref_title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt; font-weight: 600;"
        )
        cloth_ref_layout.addWidget(cloth_ref_title)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel(self.tr("Cloth brightness threshold:")))
        self._cloth_ref_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._cloth_ref_thresh_slider.setRange(0, 255)
        self._cloth_ref_thresh_slider.setValue(
            self._detection_cfg.hough.cloth_brightness_threshold
        )
        self._cloth_ref_thresh_lbl = QLabel(
            str(self._detection_cfg.hough.cloth_brightness_threshold)
        )
        self._cloth_ref_thresh_lbl.setFixedWidth(30)
        thresh_row.addWidget(self._cloth_ref_thresh_slider, 1)
        thresh_row.addWidget(self._cloth_ref_thresh_lbl)
        self._save_cloth_ref_btn = QPushButton(self.tr("Save Cloth Reference"))
        self._save_cloth_ref_btn.setObjectName("secondary")
        thresh_row.addWidget(self._save_cloth_ref_btn)
        cloth_ref_layout.addLayout(thresh_row)

        self._cloth_ref_status = QLabel(self.tr("No cloth reference saved yet."))
        self._cloth_ref_status.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;"
        )
        self._cloth_ref_status.setWordWrap(True)
        cloth_ref_layout.addWidget(self._cloth_ref_status)
        layout.addWidget(cloth_ref_frame)

        self._cloth_ref_thresh_slider.valueChanged.connect(
            self._on_cloth_ref_thresh_changed
        )
        self._save_cloth_ref_btn.clicked.connect(
            lambda: self.save_cloth_reference_requested.emit(
                self._cloth_ref_thresh_slider.value()
            )
        )

        # Detection polarity: dark dough on bright cloth vs bright dough on dark belt.
        self._dough_darker_check = QCheckBox(
            self.tr("Dough darker than background (Teig dunkler als Hintergrund)")
        )
        self._dough_darker_check.setChecked(True)
        self._dough_darker_check.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_SMALL}pt;"
        )
        layout.addWidget(self._dough_darker_check)

        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel(self.tr("Sensitivity:")))
        self._sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self._sensitivity_slider.setRange(1, 10)
        self._sensitivity_slider.setValue(5)
        self._sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sensitivity_slider.setTickInterval(1)
        self._sens_val_lbl = QLabel("5")
        self._sens_val_lbl.setFixedWidth(20)
        sens_row.addWidget(self._sensitivity_slider, 1)
        sens_row.addWidget(self._sens_val_lbl)
        layout.addLayout(sens_row)

        self._wiz_learn_status = QLabel("")
        self._wiz_learn_status.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;"
        )
        self._wiz_learn_status.setWordWrap(True)
        layout.addWidget(self._wiz_learn_status)

        def _on_sens_move(val: int) -> None:
            self._sens_val_lbl.setText(str(val))
            if self._det_canvas._frame is not None:
                self._run_detection_preview(self._det_canvas._frame)

        def _on_sens_release() -> None:
            sens = self._sensitivity_slider.value()
            nt = self._sens_to_noise_threshold(sens)
            self.noise_threshold_changed.emit(nt)

        def _on_darker_toggled(checked: bool) -> None:
            if self._det_canvas._frame is not None:
                self._run_detection_preview(self._det_canvas._frame)
            self.dough_is_darker_changed.emit(checked)

        self._sensitivity_slider.valueChanged.connect(_on_sens_move)
        self._sensitivity_slider.sliderReleased.connect(_on_sens_release)
        self._dough_darker_check.toggled.connect(_on_darker_toggled)
        return page

    @staticmethod
    def _sens_to_noise_threshold(sens: int) -> float:
        return max(0.01, min(0.99, round(0.9 - (sens - 1) * 0.0944, 3)))

    def _dispatch_detection(self) -> None:
        """Timer callback: run detection on the latest stored cloth frame."""
        if self._pending_cloth_frame is not None:
            self._run_detection_preview(self._pending_cloth_frame)

    def _run_detection_preview(self, cloth: np.ndarray) -> None:
        """``cloth`` is always the FULL (uncropped) pending cloth frame — never
        feed this an already-cropped image (``_on_sens_move``/``_on_darker_toggled``
        re-call this with ``self._det_canvas._frame``, which must stay the full
        frame across repeated calls or the crop would compound).
        """
        if self._current_profile is None:
            return
        nt = self._sens_to_noise_threshold(self._sensitivity_slider.value())
        darker = self._dough_darker_check.isChecked()
        # Use a very large fused_threshold during calibration so all detected
        # blobs are classified as "single" regardless of size.  Before "Learn
        # Reference" is clicked the profile expected_width/height are placeholder
        # values; classifying real-sized donuts as "fused" would show 0 singles
        # and keep Learn Reference disabled even when donuts are visible.
        temp = self._current_profile.model_copy(
            update={
                "noise_threshold": nt,
                "dough_is_darker": darker,
                "fused_threshold": 10.0,
            }
        )

        # FIX 1: restrict detection to the cloth ROI — without this, clutter
        # below the cloth (rag, motor) can be detected as fake "pieces" here,
        # exactly the bug this fix removes from the production path too.
        crop_region = CropRegion(
            top=self._current_cloth_crop[0], bottom=self._current_cloth_crop[1],
            left=self._current_cloth_crop[2], right=self._current_cloth_crop[3],
        )
        cropped, (cx1, cy1, _cx2, _cy2) = InspectionPipeline.apply_crop(cloth, crop_region)

        # FIX 2: dispatch by the same detection.method as production
        # (InspectionPipeline.run_cloth_tracking) so this page is a genuine
        # live A/B comparison, not a fixed blob-only preview.
        method = self._detection_cfg.method
        cropped_ref: np.ndarray | None = None  # only set/used for bg_subtract
        try:
            if method == "contour_external":
                detections, _bin, _scale = self._calibration_detector.detect_contour_external(
                    cropped, temp,
                    min_circularity=self._detection_cfg.contour_external.min_circularity,
                )
            elif method == "hough":
                h = self._detection_cfg.hough
                detections, _bin, _scale = self._calibration_detector.detect_hough(
                    cropped, temp, dp=h.dp, min_dist_px=h.min_dist_px,
                    param1=h.param1, param2=h.param2,
                    min_radius_px=h.min_radius_px, max_radius_px=h.max_radius_px,
                    gate_to_cloth=h.gate_to_cloth,
                    cloth_brightness_threshold=h.cloth_brightness_threshold,
                    downscale_factor=h.downscale_factor,
                )
            elif method == "bg_subtract":
                if (
                    self._bg_reference_cloth is not None
                    and self._bg_reference_cloth.shape[:2] == cloth.shape[:2]
                ):
                    cropped_ref, _ = InspectionPipeline.apply_crop(self._bg_reference_cloth, crop_region)
                detections, _bin, _scale = self._calibration_detector.detect_bg_subtract(
                    cropped, cropped_ref, temp,
                    threshold=self._detection_cfg.bg_subtract.threshold,
                )
            else:  # "blob" — default, exact previous behavior (now ROI-restricted)
                detections = self._calibration_detector.detect(cropped, temp)
        except Exception:
            detections = []

        # Offset crop-local detections back to full-cloth-local coordinates —
        # _det_canvas shows the FULL (uncropped) cloth frame so the operator
        # can see the crop boundary in context; see the docstring above for
        # why ``cloth`` itself must never be the cropped image.
        detections = InspectionPipeline._offset_detections(detections, cx1, cy1)

        # Update diagnostic visualizations (histogram + cloth binary/diff mask).
        # Always derived from the SAME crop used for detection above — same
        # mask-computation logic as MainWindow's "Show cloth detection mask"
        # diagnostic (see main_window._on_frame), so the two pages agree.
        try:
            if method == "bg_subtract":
                self._calib_thresh_lbl.setText(self.tr("Threshold: — (bg_subtract)"))
                mask = self._calibration_detector.compute_bg_subtract_mask(
                    cropped, cropped_ref,
                    threshold=self._detection_cfg.bg_subtract.threshold,
                )
                self._calib_hist_widget.set_data(cropped, None)
            elif method == "hough":
                # Hough produces no threshold mask of its own, so show the
                # CLOTH-REGION mask instead: the bright cloth area Hough is
                # gated to (tune cloth_brightness_threshold against this).
                # Uses compute_cloth_region_mask — the same plain brightness
                # split "Save Cloth Reference" stores, so the panel matches
                # exactly what gets saved. (The earlier compute_hough_cloth_mask
                # here left the panel blank: its wide morphological-close basis
                # collapses to all-black on a dark crop / placeholder profile.)
                h_cfg = self._detection_cfg.hough
                self._calib_thresh_lbl.setText(
                    self.tr("Cloth brightness threshold: {val}").format(
                        val=h_cfg.cloth_brightness_threshold
                    )
                )
                mask = self._calibration_detector.compute_cloth_region_mask(
                    cropped, brightness_threshold=h_cfg.cloth_brightness_threshold,
                )
                self._calib_hist_widget.set_data(cropped, h_cfg.cloth_brightness_threshold)
            else:
                fill = method == "blob" and self._detection_cfg.fill_mask_holes
                mask, thresh_val = self._calibration_detector.compute_binary_mask(
                    cropped, temp,
                    fill_holes=fill,
                    fill_holes_kernel=self._detection_cfg.fill_mask_holes_kernel,
                )
                self._calib_thresh_lbl.setText(
                    self.tr("Threshold: {val}").format(val=thresh_val)
                )
                self._calib_hist_widget.set_data(cropped, thresh_val)
            if mask is not None and mask.size > 1:
                mask_small = cv2.resize(mask, (200, 200), interpolation=cv2.INTER_NEAREST)
                qimg = QImage(mask_small.tobytes(), 200, 200, 200, QImage.Format.Format_Grayscale8)
                self._calib_mask_lbl.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

        # Belt binary mask thumbnail — read-only, shows what the belt detector
        # sees. Belt detection always uses Blob (FIX 2 only switches cloth).
        if self._pending_belt_frame is not None:
            try:
                belt_crop_region = CropRegion(
                    top=self._current_belt_crop[0], bottom=self._current_belt_crop[1],
                    left=self._current_belt_crop[2], right=self._current_belt_crop[3],
                )
                belt_cropped, _ = InspectionPipeline.apply_crop(
                    self._pending_belt_frame, belt_crop_region
                )
                if belt_cropped.size > 0 and self._current_profile is not None:
                    belt_temp = self._current_profile.model_copy(
                        update={"dough_is_darker": self._current_belt_dough_is_darker}
                    )
                    belt_mask, _ = self._calibration_detector.compute_binary_mask(
                        belt_cropped, belt_temp
                    )
                    belt_mask_small = cv2.resize(
                        belt_mask, (200, 200), interpolation=cv2.INTER_NEAREST
                    )
                    qimg_b = QImage(
                        belt_mask_small.tobytes(), 200, 200, 200,
                        QImage.Format.Format_Grayscale8,
                    )
                    self._calib_belt_mask_lbl.setPixmap(QPixmap.fromImage(qimg_b))
            except Exception:
                pass

        singles = [d for d in detections if d.label == "single"]
        n = len(singles)
        self._det_canvas.set_data(cloth, detections)
        self._calib_count_lbl.setText(
            self.tr("Detecting: {n} single piece{s}").format(
                n=n, s="s" if n != 1 else ""
            )
        )
        self._learn_btn.setEnabled(n >= 3)
        self._learn_btn.setToolTip(
            ""
            if n >= 3
            else self.tr(
                "Need at least 3 detected pieces. Currently {n}."
            ).format(n=n)
        )

    # ── navigation ───────────────────────────────────────────────────────────

    def _go_next(self) -> None:
        # Commit the departing step's settings.
        if self._current_step == 1:
            self.roi_split_x_changed.emit(self._current_roi_split_x)
        elif self._current_step == 2:
            self.transfer_line_x_changed.emit(self._current_transfer_line_x)
        elif self._current_step == 3:
            self.belt_crop_changed.emit(*self._current_belt_crop)
        elif self._current_step == 4:
            self.cloth_crop_changed.emit(*self._current_cloth_crop)
        elif self._current_step == 5:
            # Commit any in-progress slider values that weren't released.
            self.exposure_changed.emit(self._current_exposure_us)
            self.gain_changed.emit(self._current_gain)

        if self._current_step < len(self.STEPS) - 1:
            self._current_step += 1
            self._stack.setCurrentIndex(self._current_step)
            self._update_nav()
        else:
            self.completed.emit(self._collect_settings())

    def _go_back(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._stack.setCurrentIndex(self._current_step)
            self._update_nav()

    def _update_nav(self) -> None:
        self._subtitle.setText(
            self.tr("Step {n} of {total}: {name}").format(
                n=self._current_step + 1,
                total=len(self.STEPS),
                name=self.STEPS[self._current_step],
            )
        )
        self._back_btn.setEnabled(self._current_step > 0)
        self._next_btn.setText(
            self.tr("Finish")
            if self._current_step == len(self.STEPS) - 1
            else self.tr("Next")
        )

    # ── settings ─────────────────────────────────────────────────────────────

    def _collect_settings(self) -> dict:
        return {
            "rotation": self._wiz_rotation.currentData(),
            "flip_horizontal": self._wiz_flip.isChecked(),
            "roi_split_x": self._current_roi_split_x,
            "transfer_line_x": self._current_transfer_line_x,
            "exposure_us": self._current_exposure_us,
            "gain": self._current_gain,
        }

    # ── frame routing ─────────────────────────────────────────────────────────

    def set_preview_frame(self, frame: np.ndarray) -> None:
        """Route an incoming camera frame to the active step's canvas."""
        step = self._current_step
        if step == 0:
            self._wiz_orient_preview.set_frame(frame)
        elif step == 1:
            self._roi_canvas.set_frame(frame)
        elif step == 2:
            cloth = frame[:, self._current_roi_split_x :]
            if cloth.size > 0:
                self._tl_canvas.set_frame(cloth)
        elif step == 3:
            belt = frame[:, :self._current_roi_split_x]
            if belt.size > 0:
                self._belt_crop_canvas.set_frame(belt)
        elif step == 4:
            cloth = frame[:, self._current_roi_split_x :]
            if cloth.size > 0:
                self._cloth_crop_canvas.set_frame(cloth)
        elif step == 5:
            self._exp_preview.set_frame(frame)
            self._histogram.set_frame(frame)
        elif step == 6:
            belt = frame[:, :self._current_roi_split_x]
            cloth = frame[:, self._current_roi_split_x:]
            if cloth.size > 0:
                self._pending_cloth_frame = cloth
            if belt.size > 0:
                self._pending_belt_frame = belt
            # Fire throttle timer at most once per 500 ms — detection is slow.
            if not self._det_throttle_timer.isActive():
                self._det_throttle_timer.start()

    # ── public API ────────────────────────────────────────────────────────────

    def restart(self) -> None:
        self._current_step = 0
        self._stack.setCurrentIndex(0)
        self._update_nav()
        self._wiz_learn_status.setText("")
        self._calib_count_lbl.setText(self.tr("Detecting: — pieces"))
        self._learn_btn.setEnabled(False)
        self._learn_btn.setText(self.tr("Learn Reference"))

    def go_to_step(self, step: int) -> None:
        """Jump directly to a specific step (used by 'Adjust crops' shortcut)."""
        step = max(0, min(step, len(self.STEPS) - 1))
        self._current_step = step
        self._stack.setCurrentIndex(step)
        self._update_nav()

    def set_initial_orientation(self, rotation: int, flip: bool) -> None:
        """Seed step 1 without triggering apply_orientation_requested."""
        self._wiz_rotation.blockSignals(True)
        idx = self._wiz_rotation.findData(rotation)
        if idx >= 0:
            self._wiz_rotation.setCurrentIndex(idx)
        self._wiz_rotation.blockSignals(False)
        self._wiz_flip.blockSignals(True)
        self._wiz_flip.setChecked(flip)
        self._wiz_flip.blockSignals(False)

    _METHOD_LABELS = {
        "blob": "Blob",
        "contour_external": "Contour (external)",
        "hough": "Hough circles",
        "bg_subtract": "Background subtraction",
    }

    def set_detection_settings(self, detection_cfg: "_DetectionSection") -> None:
        """FIX 2: mirror AppConfig.detection onto the calibration page combo +
        label. Called by MainWindow on wizard open AND every time the method
        changes from either the Service page or this page, so both stay in
        sync. No signals emitted (this is the "receive an update" path, not
        the "request a change" path — that's _on_calib_method_changed).
        """
        self._detection_cfg = detection_cfg
        idx = self._calib_method_combo.findData(detection_cfg.method)
        if idx >= 0:
            self._calib_method_combo.blockSignals(True)
            self._calib_method_combo.setCurrentIndex(idx)
            self._calib_method_combo.blockSignals(False)
        self._calib_active_method_lbl.setText(
            self.tr("Active: {m}").format(
                m=self._METHOD_LABELS.get(detection_cfg.method, detection_cfg.method)
            )
        )
        # Mirror the row-grouping flag onto the checkbox + on/off label (receive
        # path — no signal). blockSignals so this doesn't re-emit a change.
        self._calib_row_grouping_chk.blockSignals(True)
        self._calib_row_grouping_chk.setChecked(detection_cfg.use_row_grouping)
        self._calib_row_grouping_chk.blockSignals(False)
        self._calib_row_grouping_lbl.setText(
            self.tr("Active: ON") if detection_cfg.use_row_grouping else self.tr("Active: OFF")
        )
        # Mirror grid columns and rows onto their spin boxes (receive path — no signal).
        self._grid_columns_spin.blockSignals(True)
        self._grid_columns_spin.setValue(detection_cfg.grid_columns)
        self._grid_columns_spin.blockSignals(False)
        self._grid_rows_spin.blockSignals(True)
        self._grid_rows_spin.setValue(detection_cfg.grid_rows)
        self._grid_rows_spin.blockSignals(False)
        # Seed the Cloth Reference threshold slider from the saved config so the
        # preview/save start at the persisted value (no signal — receive path).
        self._cloth_ref_thresh_slider.blockSignals(True)
        self._cloth_ref_thresh_slider.setValue(
            detection_cfg.hough.cloth_brightness_threshold
        )
        self._cloth_ref_thresh_slider.blockSignals(False)
        self._cloth_ref_thresh_lbl.setText(
            str(detection_cfg.hough.cloth_brightness_threshold)
        )
        # Re-run immediately so the live overlay/mask reflects the new method
        # without waiting for the next 500 ms timer tick.
        if self._pending_cloth_frame is not None:
            self._run_detection_preview(self._pending_cloth_frame)

    def set_bg_reference(self, image: np.ndarray | None) -> None:
        """FIX 2: mirror MainWindow._bg_reference_cloth so bg_subtract can be
        previewed here too. None means no reference captured yet (capture is
        still Service-page only; see ServiceView "Capture empty-cloth reference").
        """
        self._bg_reference_cloth = image

    def set_cloth_reference(self, mask: np.ndarray | None) -> None:
        """Mirror MainWindow._cloth_reference_mask for the saved-status label.
        Called on wizard open and after each successful save. None means no
        cloth reference saved yet (the Hough path falls back to the per-frame
        mask, see ClassicalDetector.compute_hough_cloth_mask).
        """
        self._cloth_reference_mask = mask
        if mask is not None:
            self._cloth_ref_status.setStyleSheet(
                f"color: {SUCCESS_GREEN}; font-size: {FONT_SMALL}pt;"
            )
            self._cloth_ref_status.setText(self.tr("Cloth Reference set."))
        else:
            self._cloth_ref_status.setStyleSheet(
                f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;"
            )
            self._cloth_ref_status.setText(self.tr("No cloth reference saved yet."))

    def _on_cloth_ref_thresh_changed(self, value: int) -> None:
        """Live-tune the cloth brightness threshold. Updates the Cloth Binary
        Mask panel immediately with the cloth-region mask at this threshold
        (see _update_cloth_region_preview) so the operator can dial cloth
        cleanly apart from metal/reflections before saving. Also mirrors the
        value onto the local _detection_cfg (same pattern as
        _on_calib_method_changed) so the periodic hough refresh and the
        histogram threshold line stay in sync; reconciled with the saved
        config on the next set_detection_settings (e.g. after Save).
        """
        self._cloth_ref_thresh_lbl.setText(str(value))
        self._detection_cfg = self._detection_cfg.model_copy(
            update={
                "hough": self._detection_cfg.hough.model_copy(
                    update={"cloth_brightness_threshold": value}
                )
            }
        )
        self._update_cloth_region_preview(value)

    def _update_cloth_region_preview(self, threshold: int) -> None:
        """Render the cloth-region mask at ``threshold`` into the Cloth Binary
        Mask panel — display only, and computed with the exact same
        compute_cloth_region_mask that "Save Cloth Reference" persists, so the
        live preview is precisely what gets stored. Works regardless of the
        active detection method (the slider only configures Hough gating, but
        the operator gets instant feedback either way) and is cheap (a plain
        brightness split, no detection re-run), so it's safe to call on every
        slider tick.
        """
        if self._pending_cloth_frame is None or self._current_profile is None:
            return
        crop_region = CropRegion(
            top=self._current_cloth_crop[0], bottom=self._current_cloth_crop[1],
            left=self._current_cloth_crop[2], right=self._current_cloth_crop[3],
        )
        cropped, _ = InspectionPipeline.apply_crop(self._pending_cloth_frame, crop_region)
        if cropped.size == 0:
            return
        try:
            mask = self._calibration_detector.compute_cloth_region_mask(
                cropped, brightness_threshold=threshold,
            )
            self._calib_thresh_lbl.setText(
                self.tr("Cloth brightness threshold: {val}").format(val=threshold)
            )
            self._calib_hist_widget.set_data(cropped, threshold)
            if mask is not None and mask.size > 1:
                mask_small = cv2.resize(mask, (200, 200), interpolation=cv2.INTER_NEAREST)
                qimg = QImage(
                    mask_small.tobytes(), 200, 200, 200,
                    QImage.Format.Format_Grayscale8,
                )
                self._calib_mask_lbl.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            pass

    def _on_calib_method_changed(self) -> None:
        method = self._calib_method_combo.currentData()
        if not method:
            return
        self._detection_cfg = self._detection_cfg.model_copy(update={"method": method})
        self._calib_active_method_lbl.setText(
            self.tr("Active: {m}").format(m=self._METHOD_LABELS.get(method, method))
        )
        self.detection_method_changed.emit(method)
        if self._pending_cloth_frame is not None:
            self._run_detection_preview(self._pending_cloth_frame)

    def _on_calib_row_grouping_changed(self, checked: bool) -> None:
        """Request the row-grouping flag change (per-row stop). Mirrors the
        method-combo pattern: update the local cfg, refresh the on/off label,
        and let MainWindow persist + sync the shared app-level setting."""
        self._detection_cfg = self._detection_cfg.model_copy(
            update={"use_row_grouping": checked}
        )
        self._calib_row_grouping_lbl.setText(
            self.tr("Active: ON") if checked else self.tr("Active: OFF")
        )
        self.row_grouping_changed.emit(checked)

    def set_profile_values(self, profile: ProductProfile) -> None:
        """Seed steps 2–7 with active profile values (no signals emitted)."""
        self._current_profile = profile
        self._current_roi_split_x = profile.roi_split_x
        self._current_transfer_line_x = profile.transfer_line_x
        self._current_exposure_us = profile.camera_exposure_us
        self._current_gain = profile.camera_gain

        self._roi_canvas.set_line_x(profile.roi_split_x)
        self._roi_x_lbl.setText(f"roi_split_x = {profile.roi_split_x} px")

        tl_local = max(0, profile.transfer_line_x - profile.roi_split_x)
        self._tl_canvas.set_line_x(tl_local)
        self._tl_x_lbl.setText(
            f"transfer_line_x = {profile.transfer_line_x} px (full frame)"
        )

        # SECTION 4: seed the transfer-bridge width from the profile (no signal).
        self._bridge_width_spin.blockSignals(True)
        self._bridge_width_spin.setValue(profile.transfer_bridge_width_px)
        self._bridge_width_spin.blockSignals(False)

        self._exp_slider.blockSignals(True)
        self._exp_slider.setValue(self._exposure_to_slider(profile.camera_exposure_us))
        self._exp_slider.blockSignals(False)
        self._exp_val_lbl.setText(f"{profile.camera_exposure_us} μs")

        self._gain_slider.blockSignals(True)
        self._gain_slider.setValue(int(profile.camera_gain * 10))
        self._gain_slider.blockSignals(False)
        self._gain_val_lbl.setText(f"{profile.camera_gain:.1f} dB")

        # Seed sensitivity from saved noise_threshold
        # Inverse of: nt = 0.9 - (sens-1) * 0.0944
        saved_nt = profile.noise_threshold
        sens = round(1 + (0.9 - saved_nt) / 0.0944)
        sens = max(1, min(10, sens))
        self._sensitivity_slider.blockSignals(True)
        self._sensitivity_slider.setValue(sens)
        self._sensitivity_slider.blockSignals(False)
        self._sens_val_lbl.setText(str(sens))

        self._dough_darker_check.blockSignals(True)
        self._dough_darker_check.setChecked(profile.dough_is_darker)
        self._dough_darker_check.blockSignals(False)

        # Seed crop values.
        bc = profile.belt_crop
        self._current_belt_crop = (bc.top, bc.bottom, bc.left, bc.right)
        self._belt_crop_canvas.set_crop(*self._current_belt_crop)
        self._belt_crop_lbl.setText(self._crop_text("belt", self._current_belt_crop))

        self._current_belt_dough_is_darker = profile.belt_dough_is_darker
        self._belt_darker_check.blockSignals(True)
        self._belt_darker_check.setChecked(profile.belt_dough_is_darker)
        self._belt_darker_check.blockSignals(False)

        cc = profile.cloth_crop
        self._current_cloth_crop = (cc.top, cc.bottom, cc.left, cc.right)
        self._cloth_crop_canvas.set_crop(*self._current_cloth_crop)
        self._cloth_crop_lbl.setText(self._crop_text("cloth", self._current_cloth_crop))

    def set_learn_result(self, message: str) -> None:
        is_ok = message.startswith("Learned")
        color = SUCCESS_GREEN if is_ok else "#C8102E"
        self._wiz_learn_status.setStyleSheet(
            f"color: {color}; font-size: {FONT_SMALL}pt;"
        )
        self._wiz_learn_status.setText(message)
        if is_ok:
            self._learn_btn.setText(self.tr("Learn Again"))

    # Kept for backward compatibility; no-ops in the new wizard.
    def set_camera_test_result(self, message: str) -> None:  # noqa: D401
        pass

    def set_known_profiles(self, names: list[str]) -> None:  # noqa: D401
        pass
