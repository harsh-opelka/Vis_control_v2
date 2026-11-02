"""Tests for the sidebar info block text (ui/format_info.py)."""

from __future__ import annotations

from dataclasses import replace

from viscontrol.ui.format_info import WARN_COLOR, FormatInfo, format_info_html, format_info_lines

_OK = FormatInfo(
    product="Apfelschnecken", product_ok=True, cloth="black", cloth_ok=True,
    dough_is_darker=False, fused_threshold=1.8, width_px=318, height_px=318,
    cloth_brightness_threshold=48, layout_limit=1.2, ref_name="cloth_reference_black.png",
    ref_exists=True, plc_seen=True, pending=False,
)


def _texts(info: FormatInfo) -> list[str]:
    return [t for t, _ in format_info_lines(info)]


def test_known_values() -> None:
    assert _texts(_OK) == [
        "Product: Apfelschnecken   (from PLC)",
        "Cloth: black   (PLC value 1)",
        "Dough darker: false",
        "Fused thr.: 1.8",
        "Size: 318 x 318",
        "Cloth thr.: 48",
        "Layout limit: 1.2",
        "Ref: cloth_reference_black.png",
        "Source: PLC",
    ]
    assert not any(w for _, w in format_info_lines(_OK))


def test_unknown_values_show_question_mark_in_warning_and_last_values() -> None:
    lines = format_info_lines(replace(_OK, product_ok=False, cloth_ok=False))
    assert lines[0] == ("Product: ?   (using Apfelschnecken)", True)
    assert lines[1] == ("Cloth: ?   (using black)", True)
    assert "Size: 318 x 318" in [t for t, _ in lines]  # last applied values kept
    assert WARN_COLOR in format_info_html(lines)


def test_missing_reference_is_flagged() -> None:
    lines = format_info_lines(replace(_OK, ref_exists=False))
    assert ("Ref: cloth_reference_black.png (missing)", True) in lines
    assert ("Ref: - (missing)", True) in format_info_lines(replace(_OK, ref_name=""))


def test_source_variants() -> None:
    assert _texts(replace(_OK, plc_seen=False))[-1] == "Source: default (no PLC value yet)"
    assert _texts(replace(_OK, pending=True))[-1] == "Source: pending (waiting for idle)"


def test_nothing_applied_yet() -> None:
    lines = format_info_lines(replace(_OK, product=None, cloth=None, product_ok=False, cloth_ok=False))
    assert lines[0][0].startswith("Product: ?") and lines[0][1]
