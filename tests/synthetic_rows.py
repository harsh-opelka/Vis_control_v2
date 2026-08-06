"""Shared synthetic-data helpers for piece/row tracker tests.

No Qt, no cv2, no camera, no PLC, no real clock, no sleep().
"""

from __future__ import annotations

import random

from viscontrol.core.piece_tracker import PieceTrack


def make_frames(
    rows: list, n_frames: int, jitter_px: float, dx_per_frame: float, seed: int = 0,
) -> list:
    """Build ``n_frames`` of flattened (center_x, center_y, radius) detections.

    ``rows`` is a list of rows, each a list of (center_x, center_y, radius)
    STARTING positions. Every piece drifts by ``dx_per_frame`` in X per
    frame (uniform cloth motion) and gets independent jitter of
    +/- ``jitter_px`` on center_x/center_y and +/- ``jitter_px * 0.5`` on
    radius each frame — radius jitter is deliberately smaller but nonzero,
    since it's the dominant real-world tangent_x error source (see
    BACKGROUND in the refactor spec). Deterministic for a given seed.

    Returns one flat list of (center_x, center_y, radius) tuples per frame
    (all rows combined) — the same shape RawDetection construction expects.
    """
    rng = random.Random(seed)
    pieces = [(cx, cy, r) for row in rows for (cx, cy, r) in row]
    frames: list = []
    for frame_idx in range(n_frames):
        frame = []
        for cx, cy, r in pieces:
            jx = rng.uniform(-jitter_px, jitter_px)
            jy = rng.uniform(-jitter_px, jitter_px)
            jr = rng.uniform(-jitter_px * 0.5, jitter_px * 0.5)
            frame.append((cx + dx_per_frame * frame_idx + jx, cy + jy, max(1.0, r + jr)))
        frames.append(frame)
    return frames


def make_track(
    piece_id: int, tangent_x: float, center_y: float, radius: float = 30.0,
    *, fresh_observations: int = 5, row_id: int | None = None, cycle_id: int = 0,
) -> PieceTrack:
    """Build a PieceTrack directly from a desired (smoothed) tangent_x —
    for RowTracker tests, which only care about PieceTrack's SMOOTHED
    fields, never about how they got there."""
    center_x = tangent_x + radius
    return PieceTrack(
        piece_id=piece_id, cycle_id=cycle_id,
        raw_center_x=center_x, raw_center_y=center_y, raw_radius=radius,
        center_x=center_x, center_y=center_y, radius=radius,
        tangent_x=tangent_x, prev_tangent_x=tangent_x,
        first_seen_frame=0, last_seen_frame=0,
        fresh_observations=fresh_observations, missed_frames=0, row_id=row_id,
    )


class RecordingLogger:
    """Captures every structured log line piece_tracker.py/row_tracker.py
    emit, so tests can assert on prefixes without a real loguru sink.
    Mirrors tests/test_transfer_orchestrator.py's RecordingLogger."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def _record(self, level: str, msg: str, *args) -> None:
        text = msg.format(*args) if args else msg
        self.lines.append((level, text))

    def debug(self, msg, *args) -> None: self._record("debug", msg, *args)  # noqa: E704
    def info(self, msg, *args) -> None: self._record("info", msg, *args)  # noqa: E704
    def warning(self, msg, *args) -> None: self._record("warning", msg, *args)  # noqa: E704
    def error(self, msg, *args) -> None: self._record("error", msg, *args)  # noqa: E704
    def critical(self, msg, *args) -> None: self._record("critical", msg, *args)  # noqa: E704
    def exception(self, msg, *args) -> None: self._record("error", msg, *args)  # noqa: E704

    def has(self, needle: str) -> bool:
        return any(needle in text for _, text in self.lines)
