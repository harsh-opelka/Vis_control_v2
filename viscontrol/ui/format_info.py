"""Text for the sidebar's read-only product/cloth info block (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

WARN_COLOR = "#E8892B"


@dataclass(frozen=True)
class FormatInfo:
    """Snapshot of the settings ACTUALLY applied at runtime plus switch state."""

    product: str | None          # applied product (None = nothing applied yet)
    product_ok: bool             # False = last PLC product was unknown/unreadable
    cloth: str | None            # "white" / "black" / None
    cloth_ok: bool
    dough_is_darker: bool
    fused_threshold: float
    width_px: int
    height_px: int
    cloth_brightness_threshold: int
    layout_limit: float          # layout_quality.min_separation_ratio as applied
    ref_name: str               # file name only ("" = none configured)
    ref_exists: bool
    plc_seen: bool               # a PLC value has been received
    pending: bool                # a different valid target is waiting to be applied


_CLOTH_PLC_VALUE = {"white": "0", "black": "1"}


def format_info_lines(info: FormatInfo) -> list[tuple[str, bool]]:
    """Return ``[(text, is_warning), ...]`` — one short line each (<= ~36 chars)."""
    product = info.product or "?"
    cloth = info.cloth or "?"
    if info.product_ok:
        product_line = (f"Product: {product}   (from PLC)", False)
    else:
        product_line = (f"Product: ?   (using {product})", True)
    if info.cloth_ok:
        cloth_line = (f"Cloth: {cloth}   (PLC value {_CLOTH_PLC_VALUE.get(cloth, '?')})", False)
    else:
        cloth_line = (f"Cloth: ?   (using {cloth})", True)
    if info.ref_name:
        ref_line = (
            f"Ref: {info.ref_name}" + ("" if info.ref_exists else " (missing)"),
            not info.ref_exists,
        )
    else:
        ref_line = ("Ref: - (missing)", True)
    if info.pending:
        source = "pending (waiting for idle)"
    elif info.plc_seen:
        source = "PLC"
    else:
        source = "default (no PLC value yet)"
    return [
        product_line,
        cloth_line,
        (f"Dough darker: {str(info.dough_is_darker).lower()}", False),
        (f"Fused thr.: {info.fused_threshold:g}", False),
        (f"Size: {info.width_px} x {info.height_px}", False),
        (f"Cloth thr.: {info.cloth_brightness_threshold}", False),
        (f"Layout limit: {info.layout_limit:g}", False),
        ref_line,
        (f"Source: {source}", info.pending),
    ]


def format_info_html(lines: list[tuple[str, bool]]) -> str:
    return "<br>".join(
        f'<span style="color:{WARN_COLOR}">{escape(t)}</span>' if warn else escape(t)
        for t, warn in lines
    )
