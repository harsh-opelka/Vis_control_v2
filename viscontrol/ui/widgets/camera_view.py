"""Camera view widget: shows the current ROI image with detection overlays.

Used twice in the InferenceView: once for the belt ROI, once for the cloth
ROI. Overlay color/shape varies by detection ``label``; the cloth ROI also
draws the configured transfer line as a yellow dashed vertical line.

All drawing happens in :meth:`paintEvent` against the scaled pixmap so the
overlays stay aligned even when the user resizes the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QSizePolicy, QVBoxLayout, QWidget, QLabel

from viscontrol.detection.base import Detection
from viscontrol.ui.theme import (
    BORDER,
    CARD_RADIUS,
    FONT_NORMAL,
    FONT_SMALL,
    OVERLAY_COLUMN_FUSED,
    OVERLAY_ROW_FUSED,
    OVERLAY_SINGLE,
    OVERLAY_UNKNOWN,
    SUCCESS_GREEN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TRANSFER_LINE,
)


@dataclass
class CameraViewState:
    image: np.ndarray | None = None
    detections: list[Detection] | None = None
    transfer_line_local_x: int | None = None  # in image-local coords; None to hide
    px_per_mm: float = 1.0
    highlight_centroids: list[tuple[float, float]] | None = None  # bright tracking glow
    crop_rect: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2) in image-local px
    debug_mask: np.ndarray | None = None  # binary mask thumbnail (diagnostics only)
    tripwire_half_width_px: int = 0     # >0 to draw the tripwire band around transfer line
    tripwire_occupied: bool = False     # True → highlight band in warning color
    # SECTION 4: transfer bridge — a configurable-width band centered on the
    # transfer line; a piece is "at the transfer point" once its leading edge
    # overlaps it. >0 draws the band (cloth-local px).
    transfer_bridge_width_px: int = 0
    # SECTION 3: active detection band (x_left, x_right) in image-local coords;
    # pieces outside it don't drive stops. None = not drawn.
    detection_band: tuple[int, int] | None = None
    # SECTION 6: leading-edge centroids of the CURRENT/front row, drawn in a
    # distinct colour so the operator sees the grouped row.
    current_row_centroids: list[tuple[float, float]] | None = None
    # DIAGNOSTIC ONLY (two-rows-at-once investigation): column-sum dough profile
    # along the travel direction. row_profile_x_offset/scale map a profile
    # index back to image-local x: img_x = row_profile_x_offset + i / scale.
    row_profile: np.ndarray | None = None
    row_profile_x_offset: int = 0
    row_profile_scale: float = 1.0
    # Row grouping (USE_ROW_GROUPING). row_lines = travel-direction (x) positions
    # of each detected row's row-line, in image-local coords; drawn as vertical
    # lines and labelled with row_count ("Rows: N"). None when the toggle is off.
    row_lines: list[float] | None = None
    row_count: int | None = None
    # Detection zone outer boundary (approach side), cloth-local x. Drawn as a
    # distinct orange dashed line with "det.zone" label. None = not drawn.
    detection_zone_outer_x: int | None = None
    # Grid-aware stop visualization (USE_ROW_GROUPING=ON):
    #   grid_row_assignments — detection index → row number (1=front, 2=second, 0=other)
    #   grid_ref_tangent_x   — cloth-local x of the stop reference (trailing Row-1 tangent)
    #   grid_label           — informational text: "Row 1: N | Ref: Xpx"
    grid_row_assignments: dict[int, int] | None = None
    grid_ref_tangent_x: float | None = None
    grid_label: str | None = None


class _Canvas(QWidget):
    """Inner widget that actually paints the image + overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = CameraViewState()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background-color: #1A1A1A; border-radius: {CARD_RADIUS}px;")

    def set_state(self, state: CameraViewState) -> None:
        self._state = state
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        p.fillRect(rect, QColor("#1A1A1A"))

        state = self._state
        if state.image is None:
            p.setPen(QColor(TEXT_SECONDARY))
            font = QFont()
            font.setPointSize(FONT_NORMAL)
            p.setFont(font)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "no signal")
            return

        # Convert numpy array to QImage and scale to widget while preserving aspect.
        img = state.image
        h, w = img.shape[:2]
        if img.ndim == 2:
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img.tobytes(), w, h, w * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Center it.
        dx = (rect.width() - pixmap.width()) // 2
        dy = (rect.height() - pixmap.height()) // 2
        p.drawPixmap(dx, dy, pixmap)

        # Scale factor from image coords to drawn pixmap.
        sx = pixmap.width() / max(w, 1)
        sy = pixmap.height() / max(h, 1)

        # Tripwire band: semi-transparent fill over the transfer-line strip.
        # Drawn before the transfer line so the dashed line appears on top.
        if (
            state.transfer_line_local_x is not None
            and 0 <= state.transfer_line_local_x < w
            and state.tripwire_half_width_px > 0
        ):
            band_color = (
                QColor(200, 80, 0, 80) if state.tripwire_occupied
                else QColor(250, 199, 117, 28)
            )
            bx = int(dx + (state.transfer_line_local_x - state.tripwire_half_width_px) * sx)
            bw = max(2, int(2 * state.tripwire_half_width_px * sx))
            p.fillRect(bx, dy, bw, pixmap.height(), band_color)

        # SECTION 3: detection band — faint outlined region near the transfer
        # line; only pieces whose leading edge is inside it drive stops.
        if state.detection_band is not None:
            bx0 = int(dx + max(0, state.detection_band[0]) * sx)
            bx1 = int(dx + min(w, state.detection_band[1]) * sx)
            if bx1 > bx0:
                p.fillRect(bx0, dy, bx1 - bx0, pixmap.height(), QColor(0, 200, 255, 16))
                pen = QPen(QColor(0, 200, 255, 120))
                pen.setWidth(1)
                pen.setStyle(Qt.PenStyle.DotLine)
                p.setPen(pen)
                p.drawLine(bx0, dy, bx0, dy + pixmap.height())
                p.drawLine(bx1, dy, bx1, dy + pixmap.height())

        # FIX 2: transfer bridge = detection band (unified). Drawn before the
        # dashed transfer line so the line sits on top. Both edges are solid
        # bright lines so the operator sees clearly how wide the zone is.
        if (
            state.transfer_line_local_x is not None
            and 0 <= state.transfer_line_local_x < w
            and state.transfer_bridge_width_px > 0
        ):
            half = state.transfer_bridge_width_px / 2.0
            gbx = int(dx + (state.transfer_line_local_x - half) * sx)
            gbw = max(2, int(state.transfer_bridge_width_px * sx))
            bridge_fill = (
                QColor(0, 229, 160, 110) if state.tripwire_occupied
                else QColor(0, 229, 160, 55)
            )
            p.fillRect(gbx, dy, gbw, pixmap.height(), bridge_fill)
            # Edge lines: 3 px thick so both boundaries are clearly visible.
            pen = QPen(QColor(0, 229, 160))
            pen.setWidth(3)
            pen.setStyle(Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawLine(gbx, dy, gbx, dy + pixmap.height())
            p.drawLine(gbx + gbw, dy, gbx + gbw, dy + pixmap.height())
            # Width label at the top of the left edge so it's always readable.
            font = QFont()
            font.setPointSize(max(6, FONT_SMALL - 1))
            p.setFont(font)
            p.setPen(QColor(0, 229, 160))
            p.drawText(gbx + 4, dy + 14, f"{state.transfer_bridge_width_px}px")

        # Detection zone outer boundary: orange dashed line on the approach side
        # showing exactly how far from the bridge Hough actually runs.
        if state.detection_zone_outer_x is not None and 0 <= state.detection_zone_outer_x < w:
            x_zone = int(dx + state.detection_zone_outer_x * sx)
            pen = QPen(QColor(255, 140, 0))   # orange — distinct from bridge (green)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(x_zone, dy, x_zone, dy + pixmap.height())
            font = QFont()
            font.setPointSize(max(6, FONT_SMALL - 1))
            p.setFont(font)
            p.setPen(QColor(255, 140, 0))
            p.drawText(x_zone + 3, dy + 28, "det.zone")

        # Transfer line.
        if state.transfer_line_local_x is not None and 0 <= state.transfer_line_local_x < w:
            line_color = QColor(200, 80, 0) if state.tripwire_occupied else QColor(TRANSFER_LINE)
            pen = QPen(line_color)
            pen.setWidth(3)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            x_px = int(dx + state.transfer_line_local_x * sx)
            p.drawLine(x_px, dy, x_px, dy + pixmap.height())

        # Detections — row_num drives grid row coloring (1=magenta, 2=cyan, 0=default).
        _row_asgn = state.grid_row_assignments or {}
        for i, det in enumerate(state.detections or []):
            self._draw_detection(p, det, dx, dy, sx, sy, state.px_per_mm,
                                 row_num=_row_asgn.get(i, 0))

        # Grid stop reference tangent: red full-height line at the Row-1 trailing
        # piece's left tangent — the piece whose crossing fires StopTuchabzug.
        if state.grid_ref_tangent_x is not None:
            ref_x = int(dx + state.grid_ref_tangent_x * sx)
            ref_pen = QPen(QColor(255, 60, 60))
            ref_pen.setWidth(3)
            ref_pen.setStyle(Qt.PenStyle.SolidLine)
            p.setPen(ref_pen)
            p.drawLine(ref_x, dy - 6, ref_x, dy + pixmap.height() + 6)

        # Grid label: "Row 1: N | Ref: Xpx" — drawn in yellow-gold at top-left.
        if state.grid_label:
            font = QFont()
            font.setPointSize(FONT_SMALL)
            font.setBold(True)
            p.setFont(font)
            p.setPen(QColor(255, 220, 0))
            p.drawText(dx + 8, dy + 40, state.grid_label)

        # Front-row highlights (cloth tracking).
        for cx, cy in state.highlight_centroids or []:
            self._draw_highlight(p, cx, cy, dx, dy, sx, sy)

        # Crop overlay: dim outside-crop area + thin border.
        if state.crop_rect is not None:
            self._draw_crop_overlay(p, state.crop_rect, dx, dy, sx, sy, pixmap)

        # --- DIAGNOSTIC: row-profile curve (two-rows-at-once investigation) ---
        # Translucent band + curve along the bottom of the image. Read-only
        # visualization of state.row_profile — does not affect detection or
        # the tripwire. Delete this block to remove the overlay entirely.
        if state.row_profile is not None and state.row_profile.size > 1:
            self._draw_row_profile(p, state, dx, dy, sx, pixmap)
        # --- END DIAGNOSTIC ---

        # Debug mask thumbnail: bottom-right corner of the image area.
        if state.debug_mask is not None:
            mh, mw = state.debug_mask.shape[:2]
            qimg_m = QImage(
                state.debug_mask.tobytes(), mw, mh, mw, QImage.Format.Format_Grayscale8
            )
            thumb = min(160, max(60, rect.width() // 5))
            pix_m = QPixmap.fromImage(qimg_m).scaled(
                thumb, thumb,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            mx = dx + pixmap.width() - pix_m.width() - 4
            my = dy + pixmap.height() - pix_m.height() - 4
            p.fillRect(mx - 2, my - 2, pix_m.width() + 4, pix_m.height() + 4,
                       QColor("#00C8FF"))
            p.drawPixmap(mx, my, pix_m)
            font = QFont()
            font.setPointSize(max(6, FONT_SMALL - 2))
            p.setFont(font)
            p.setPen(QColor("#00C8FF"))
            p.drawText(mx, my - 3, "bin.mask")

    @staticmethod
    def _color_for(label: str) -> str:
        return {
            "single": OVERLAY_SINGLE,
            "row_fused": OVERLAY_ROW_FUSED,
            "column_fused": OVERLAY_COLUMN_FUSED,
            "unknown": OVERLAY_UNKNOWN,
        }.get(label, OVERLAY_UNKNOWN)

    def _draw_detection(
        self,
        p: QPainter,
        det: Detection,
        dx: int,
        dy: int,
        sx: float,
        sy: float,
        px_per_mm: float,
        row_num: int = 0,
    ) -> None:
        cx, cy = det.centroid
        x = dx + cx * sx
        y = dy + cy * sy
        radius_px = (max(det.width_px, det.height_px) / 2) * max(sx, sy)
        # Row color overrides label color when grid grouping is active.
        if row_num == 1:
            color = QColor("#FF2BD6")   # magenta — Row 1 (front row)
        elif row_num == 2:
            color = QColor("#00E5FF")   # cyan — Row 2
        else:
            color = QColor(self._color_for(det.label))

        if det.label == "single":
            pen = QPen(color)
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(x, y), radius_px, radius_px)
        elif det.label == "row_fused":
            pen = QPen(color)
            pen.setWidth(5)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(x, y), radius_px, radius_px)
            mm = (max(det.width_px, det.height_px) / max(px_per_mm, 0.001))
            p.setPen(color)
            font = QFont()
            font.setPointSize(FONT_SMALL)
            font.setBold(True)
            p.setFont(font)
            p.drawText(
                QPointF(x + radius_px + 4, y),
                f"FUSED {mm:.0f} mm",
            )
        elif det.label == "column_fused":
            pen = QPen(color)
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(x, y), radius_px, radius_px)
        elif det.label == "unknown":
            pen = QPen(color)
            pen.setWidth(3)
            p.setPen(pen)
            font = QFont()
            font.setPointSize(FONT_NORMAL + 6)
            font.setBold(True)
            p.setFont(font)
            p.drawText(
                QRectF(x - 16, y - 16, 32, 32),
                Qt.AlignmentFlag.AlignCenter,
                "?",
            )

        # Left-side tangent: vertical line at center_x − radius, length = diameter.
        # Row 1 drawn in magenta; all others in yellow-gold.
        r_img = max(det.width_px or 0.0, det.height_px or 0.0) / 2.0
        if r_img > 0:
            x_tang = dx + (cx - r_img) * sx
            tang_color = QColor("#FF2BD6") if row_num == 1 else QColor(255, 220, 0)
            tang_pen = QPen(tang_color)
            tang_pen.setWidth(2)
            p.setPen(tang_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(
                QPointF(x_tang, dy + (cy - r_img) * sy),
                QPointF(x_tang, dy + (cy + r_img) * sy),
            )

    @staticmethod
    def _draw_crop_overlay(
        p: QPainter,
        crop_rect: tuple[int, int, int, int],
        dx: int,
        dy: int,
        sx: float,
        sy: float,
        pixmap: "QPixmap",
    ) -> None:
        """Dim the area outside the active crop window and draw a thin border."""
        ix1, iy1, ix2, iy2 = crop_rect
        # Widget coords of the crop rectangle
        wx1 = int(dx + ix1 * sx)
        wy1 = int(dy + iy1 * sy)
        wx2 = int(dx + ix2 * sx)
        wy2 = int(dy + iy2 * sy)
        img_left = dx
        img_top = dy
        img_right = dx + pixmap.width()
        img_bot = dy + pixmap.height()

        dim = QColor(0, 0, 0, 110)
        # Top strip
        if wy1 > img_top:
            p.fillRect(img_left, img_top, pixmap.width(), wy1 - img_top, dim)
        # Bottom strip
        if wy2 < img_bot:
            p.fillRect(img_left, wy2, pixmap.width(), img_bot - wy2, dim)
        # Left strip (between crop top/bottom lines)
        if wx1 > img_left:
            p.fillRect(img_left, wy1, wx1 - img_left, wy2 - wy1, dim)
        # Right strip
        if wx2 < img_right:
            p.fillRect(wx2, wy1, img_right - wx2, wy2 - wy1, dim)

        # Thin cyan border around the active crop window
        pen = QPen(QColor("#00C8FF"))
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(wx1, wy1, wx2 - wx1, wy2 - wy1)

    def _draw_highlight(
        self,
        p: QPainter,
        cx: float,
        cy: float,
        dx: int,
        dy: int,
        sx: float,
        sy: float,
    ) -> None:
        x = dx + cx * sx
        y = dy + cy * sy
        color = QColor(OVERLAY_SINGLE)
        color.setAlpha(160)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(QPointF(x, y), 10, 10)

    @staticmethod
    def _draw_row_lines(
        p: QPainter,
        state: "CameraViewState",
        dx: int,
        dy: int,
        sx: float,
        w: int,
        pixmap: "QPixmap",
    ) -> None:
        """Row grouping overlay: a solid vertical line at each row's row-line
        (the transfer line is vertical, so rows run perpendicular to travel and
        their row-lines are vertical too), plus a "Rows: N" count badge.

        Display only — driven by CameraViewState.row_lines / row_count, which
        MainWindow populates from RowLineTracker. Does not affect detection,
        the tripwire, or firing.
        """
        line_color = QColor("#00E5A0")  # distinct teal, separate from transfer line
        pen = QPen(line_color)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        p.setPen(pen)
        for rx in state.row_lines or []:
            if not (0 <= rx < w):
                continue
            x_px = int(dx + rx * sx)
            p.drawLine(x_px, dy, x_px, dy + pixmap.height())

        count = state.row_count if state.row_count is not None else len(state.row_lines or [])
        font = QFont()
        font.setPointSize(max(8, FONT_NORMAL))
        font.setBold(True)
        p.setFont(font)
        p.setPen(line_color)
        p.drawText(dx + 8, dy + 20, f"Rows: {count}")

    @staticmethod
    def _draw_row_profile(
        p: QPainter,
        state: "CameraViewState",
        dx: int,
        dy: int,
        sx: float,
        pixmap: "QPixmap",
    ) -> None:
        """DIAGNOSTIC ONLY: draw the column-sum dough profile as a curve along
        the bottom of the image (two-rows-at-once investigation)."""
        profile = state.row_profile
        peak = float(profile.max())
        if peak <= 0:
            return
        band_h = max(24, int(pixmap.height() * 0.18))
        base_y = dy + pixmap.height()
        top_y = base_y - band_h
        p.fillRect(dx, top_y, pixmap.width(), band_h, QColor(0, 0, 0, 100))

        pen = QPen(QColor("#00C8FF"))
        pen.setWidth(2)
        p.setPen(pen)
        prev_pt: QPointF | None = None
        for i in range(profile.shape[0]):
            img_x = state.row_profile_x_offset + i / max(state.row_profile_scale, 1e-6)
            x_px = dx + img_x * sx
            norm = min(1.0, float(profile[i]) / peak)
            pt = QPointF(x_px, base_y - norm * band_h)
            if prev_pt is not None:
                p.drawLine(prev_pt, pt)
            prev_pt = pt

        p.setPen(QColor("#00C8FF"))
        font = QFont()
        font.setPointSize(max(6, FONT_SMALL - 2))
        p.setFont(font)
        p.drawText(dx + 4, top_y + 12, "row profile (DIAG)")


class CameraView(QFrame):
    """Captioned camera view (title + canvas + caption)."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        # Expand to fill available space so both belt and cloth views are equal width.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._title = QLabel(title)
        self._title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )

        # State label shown between the title and the image: e.g. "INSPECTING" /
        # "IDLE" for belt, or the RowPhase name for cloth.  Starts empty.
        self._state_label = QLabel("")
        self._state_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt; font-weight: 600;"
        )

        self._canvas = _Canvas()
        self._caption = QLabel("")
        self._caption.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: {FONT_SMALL}pt;")

        layout.addWidget(self._title)
        layout.addWidget(self._state_label)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._caption)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_caption(self, caption: str) -> None:
        self._caption.setText(caption)

    def set_state_label(self, text: str, *, active: bool = False) -> None:
        """Update the small status line above the image.

        ``active=True`` renders in green; ``False`` renders in the standard
        secondary-text gray.
        """
        self._state_label.setText(text)
        color = SUCCESS_GREEN if active else TEXT_SECONDARY
        self._state_label.setStyleSheet(
            f"color: {color}; font-size: {FONT_SMALL}pt; font-weight: 600;"
        )

    def set_state(self, state: CameraViewState) -> None:
        self._canvas.set_state(state)


class FramePreview(QWidget):
    """Compact live-preview: displays a scaled numpy frame with no title or caption.

    Used in the SERVICE orientation panel and Installation Wizard step 3 so the
    operator can see the camera feed with the currently-selected rotation applied
    before committing the change.
    """

    def __init__(
        self,
        *,
        width: int = 320,
        height: int = 240,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = _Canvas(self)
        layout.addWidget(self._canvas)
        self.setFixedSize(width, height)

    def set_frame(self, frame: np.ndarray) -> None:
        self._canvas.set_state(CameraViewState(image=frame))
