"""Generate synthetic dough images for MockCamera and pipeline tests.

Layout matches the production geometry described in the spec: a full frame is
roughly square-ish, the belt is on the left, the cloth (with the transfer
line) is on the right. We sit at the same coordinate system as the Default
profile so MockCamera frames "just work" with the default configuration.

Scenarios:
    clean_01..03        rows of perfectly spaced pieces on the belt
    row_fused_01..02    one vertical merge (two pieces touching along row axis)
    column_fused_01     one horizontal merge (informational, not fault)
    unknown_01          one piece with bizarre geometry
    cloth_pre_line      cloth full of pieces, none crossing transfer_line_x yet
    cloth_at_line       cloth pieces straddling transfer_line_x (triggers stop)
    empty_belt          empty belt (used by FAULT self-clear tests)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# Geometry — matches config/default.yaml's Default profile.
W = 3200
H = 2200
ROI_SPLIT_X = 1768
TRANSFER_LINE_X = 2400
PIECE_RADIUS = 145         # ~290 px wide pieces matches expected_width_px = 290
ROW_PITCH = 360            # vertical distance between piece centroids
COL_PITCH = 360            # horizontal distance between piece centroids
BELT_BACKGROUND = 35       # dark grey
CLOTH_BACKGROUND = 50      # slightly lighter
PIECE_VALUE = 215          # bright dough


def _fresh_frame() -> np.ndarray:
    frame = np.full((H, W), BELT_BACKGROUND, dtype=np.uint8)
    frame[:, ROI_SPLIT_X:] = CLOTH_BACKGROUND
    return frame


def _draw_piece(frame: np.ndarray, center: tuple[int, int], radius: int = PIECE_RADIUS) -> None:
    cv2.circle(frame, center, radius, PIECE_VALUE, thickness=-1)


def _draw_belt_row(frame: np.ndarray, x: int, rng: random.Random | None = None) -> None:
    """Draw a vertical row of pieces in the belt ROI at column ``x``."""
    rng = rng or random.Random(0)
    y_start = 200
    while y_start + PIECE_RADIUS < H - 100:
        jitter = rng.randint(-5, 5)
        _draw_piece(frame, (x, y_start + jitter))
        y_start += ROW_PITCH


def _draw_cloth_grid(
    frame: np.ndarray, x_start: int, x_end: int, rng: random.Random | None = None
) -> None:
    rng = rng or random.Random(1)
    x = x_start
    while x + PIECE_RADIUS < x_end:
        y_start = 200
        while y_start + PIECE_RADIUS < H - 100:
            _draw_piece(frame, (x + rng.randint(-3, 3), y_start + rng.randint(-3, 3)))
            y_start += ROW_PITCH
        x += COL_PITCH


# ---------- scenarios ----------


def scenario_clean(seed: int, name: str, out_dir: Path) -> None:
    rng = random.Random(seed)
    frame = _fresh_frame()
    # Three full rows on the belt, ending near the right edge of the belt ROI.
    for x in (ROI_SPLIT_X - 800, ROI_SPLIT_X - 440, ROI_SPLIT_X - 80):
        _draw_belt_row(frame, x, rng)
    # Cloth full of pieces, none past the transfer line.
    _draw_cloth_grid(frame, ROI_SPLIT_X + 100, TRANSFER_LINE_X - PIECE_RADIUS - 50, rng)
    cv2.imwrite(str(out_dir / f"{name}.png"), frame)


def scenario_row_fused(seed: int, name: str, out_dir: Path) -> None:
    rng = random.Random(seed)
    frame = _fresh_frame()
    for x in (ROI_SPLIT_X - 800, ROI_SPLIT_X - 440):
        _draw_belt_row(frame, x, rng)
    # Newest row at ROI_SPLIT_X - 80; one fused pair.
    x = ROI_SPLIT_X - 80
    y = 200
    fused_index = 2  # which slot to merge
    slot = 0
    while y + PIECE_RADIUS < H - 100:
        if slot == fused_index:
            # Vertical pill that's ~2x the height of a single piece.
            cv2.ellipse(
                frame, (x, y + ROW_PITCH // 2),
                (PIECE_RADIUS, ROW_PITCH), 0, 0, 360, PIECE_VALUE, -1,
            )
            y += 2 * ROW_PITCH
        else:
            _draw_piece(frame, (x, y))
            y += ROW_PITCH
        slot += 1
    _draw_cloth_grid(frame, ROI_SPLIT_X + 100, TRANSFER_LINE_X - PIECE_RADIUS - 50, rng)
    cv2.imwrite(str(out_dir / f"{name}.png"), frame)


def scenario_column_fused(seed: int, name: str, out_dir: Path) -> None:
    rng = random.Random(seed)
    frame = _fresh_frame()
    # Two rows clean, third row has a horizontal pill (column_fused).
    for x in (ROI_SPLIT_X - 800, ROI_SPLIT_X - 440):
        _draw_belt_row(frame, x, rng)
    x = ROI_SPLIT_X - 80
    y = 200 + 2 * ROW_PITCH
    cv2.ellipse(
        frame, (x, y),
        (int(PIECE_RADIUS * 2.0), PIECE_RADIUS), 0, 0, 360, PIECE_VALUE, -1,
    )
    # Other pieces normal.
    for slot_y in range(200, H - 100, ROW_PITCH):
        if abs(slot_y - y) < PIECE_RADIUS:
            continue
        _draw_piece(frame, (x, slot_y))
    _draw_cloth_grid(frame, ROI_SPLIT_X + 100, TRANSFER_LINE_X - PIECE_RADIUS - 50, rng)
    cv2.imwrite(str(out_dir / f"{name}.png"), frame)


def scenario_unknown(seed: int, name: str, out_dir: Path) -> None:
    rng = random.Random(seed)
    frame = _fresh_frame()
    for x in (ROI_SPLIT_X - 800, ROI_SPLIT_X - 440):
        _draw_belt_row(frame, x, rng)
    x = ROI_SPLIT_X - 80
    # A weird L-shape: tall + wide = unknown.
    cv2.ellipse(
        frame, (x, 400),
        (int(PIECE_RADIUS * 1.8), int(PIECE_RADIUS * 1.8)), 0, 0, 360, PIECE_VALUE, -1,
    )
    for slot_y in (800, 1200, 1600, 2000):
        _draw_piece(frame, (x, slot_y))
    _draw_cloth_grid(frame, ROI_SPLIT_X + 100, TRANSFER_LINE_X - PIECE_RADIUS - 50, rng)
    cv2.imwrite(str(out_dir / f"{name}.png"), frame)


def scenario_cloth_at_line(seed: int, name: str, out_dir: Path) -> None:
    rng = random.Random(seed)
    frame = _fresh_frame()
    for x in (ROI_SPLIT_X - 800, ROI_SPLIT_X - 440, ROI_SPLIT_X - 80):
        _draw_belt_row(frame, x, rng)
    # Cloth grid pushed forward — front row is at the transfer line.
    _draw_cloth_grid(frame, ROI_SPLIT_X + 100, TRANSFER_LINE_X - PIECE_RADIUS - 50, rng)
    # Plus one extra row right ON the line.
    for y in range(200, H - 100, ROW_PITCH):
        _draw_piece(frame, (TRANSFER_LINE_X + 30, y))
    cv2.imwrite(str(out_dir / f"{name}.png"), frame)


def scenario_empty_belt(seed: int, name: str, out_dir: Path) -> None:
    rng = random.Random(seed)
    frame = _fresh_frame()
    _draw_cloth_grid(frame, ROI_SPLIT_X + 100, TRANSFER_LINE_X - PIECE_RADIUS - 50, rng)
    cv2.imwrite(str(out_dir / f"{name}.png"), frame)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "test_images",
        help="output directory (default: assets/test_images/)",
    )
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    scenario_clean(seed=1, name="01_clean", out_dir=out)
    scenario_clean(seed=2, name="02_clean", out_dir=out)
    scenario_clean(seed=3, name="03_clean", out_dir=out)
    scenario_row_fused(seed=10, name="10_row_fused", out_dir=out)
    scenario_row_fused(seed=11, name="11_row_fused", out_dir=out)
    scenario_column_fused(seed=20, name="20_column_fused", out_dir=out)
    scenario_unknown(seed=30, name="30_unknown", out_dir=out)
    scenario_cloth_at_line(seed=40, name="40_cloth_at_line", out_dir=out)
    scenario_empty_belt(seed=50, name="50_empty_belt", out_dir=out)

    print(f"Wrote synthetic test images to {out}")
    for p in sorted(out.glob("*.png")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
