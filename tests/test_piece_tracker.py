"""Tests for viscontrol.core.piece_tracker (PieceTracker).

No Qt, no cv2, no camera, no PLC, no real clock, no sleep().
"""

from __future__ import annotations

import random
import statistics

from viscontrol.core.config import AppConfig
from viscontrol.core.piece_tracker import PieceTracker, RawDetection
from tests.synthetic_rows import make_frames


def _cfg():
    return AppConfig().piece_tracker


def _dets(frame: list, frame_id: int) -> list:
    return [
        RawDetection(center_x=cx, center_y=cy, radius=r, source_frame_id=frame_id, det_index=i)
        for i, (cx, cy, r) in enumerate(frame)
    ]


# ---------------------------------------------------------------------------
# P1 — stable_ids_stationary
# ---------------------------------------------------------------------------


def test_P1_stable_ids_stationary() -> None:
    tracker = PieceTracker(_cfg())
    rows = [[(500.0 + i * 100.0, 200.0, 30.0) for i in range(7)]]
    frames = make_frames(rows, n_frames=30, jitter_px=15.0, dx_per_frame=0.0, seed=1)

    creates_per_frame = []
    drops_per_frame = []
    result = None
    for frame_id, frame in enumerate(frames):
        result = tracker.update(_dets(frame, frame_id), frame_id)
        creates_per_frame.append(result.created)
        drops_per_frame.append(result.dropped)

    assert len(result.tracks) == 7
    assert len(creates_per_frame[0]) == 7, "all 7 pieces must be created on frame 1"
    assert all(len(c) == 0 for c in creates_per_frame[1:]), "no creates after frame 1"
    assert all(len(d) == 0 for d in drops_per_frame), "no drops for stationary pieces"
    assert len({t.piece_id for t in result.tracks}) == 7


# ---------------------------------------------------------------------------
# P2 — one_to_one
# ---------------------------------------------------------------------------


def test_P2_one_to_one() -> None:
    tracker = PieceTracker(_cfg())
    rows = [
        [(500.0 + i * 90.0, 200.0 + j * 150.0, 28.0) for i in range(4)]
        for j in range(3)
    ]
    frames = make_frames(rows, n_frames=100, jitter_px=15.0, dx_per_frame=-3.0, seed=2)

    for frame_id, frame in enumerate(frames):
        result = tracker.update(_dets(frame, frame_id), frame_id)
        matched_piece_ids = [m[0] for m in result.matched]
        matched_det_idx = [m[1] for m in result.matched]
        assert len(matched_piece_ids) == len(set(matched_piece_ids)), "a piece matched twice this frame"
        assert len(matched_det_idx) == len(set(matched_det_idx)), "a detection matched twice this frame"


# ---------------------------------------------------------------------------
# P3 — shuffle_invariance
# ---------------------------------------------------------------------------


def _run_shuffled(frames: list, shuffle_seed: int) -> dict:
    tracker = PieceTracker(_cfg())
    rng = random.Random(shuffle_seed)
    mapping: dict[int, tuple[float, float]] = {}
    for frame_id, frame in enumerate(frames):
        order = list(range(len(frame)))
        rng.shuffle(order)
        dets = [
            RawDetection(
                center_x=frame[oi][0], center_y=frame[oi][1], radius=frame[oi][2],
                source_frame_id=frame_id, det_index=pos,
            )
            for pos, oi in enumerate(order)
        ]
        result = tracker.update(dets, frame_id)
        for t in result.tracks:
            mapping[t.piece_id] = (round(t.center_x, 6), round(t.center_y, 6))
    return mapping


def test_P3_shuffle_invariance() -> None:
    rows = [
        [(500.0 + i * 90.0, 200.0 + j * 150.0, 28.0) for i in range(3)]
        for j in range(2)
    ]
    frames = make_frames(rows, n_frames=15, jitter_px=15.0, dx_per_frame=-2.0, seed=3)

    baseline = _run_shuffled(frames, shuffle_seed=999)
    for seed in range(20):
        result = _run_shuffled(frames, shuffle_seed=seed)
        assert result == baseline, f"shuffle seed={seed} produced a different piece_id mapping"


# ---------------------------------------------------------------------------
# P4 — smoothing_reduces_tangent_variance
# ---------------------------------------------------------------------------


def test_P4_smoothing_reduces_tangent_variance() -> None:
    tracker = PieceTracker(_cfg())
    rows = [[(600.0, 200.0, 30.0)]]
    frames = make_frames(rows, n_frames=40, jitter_px=15.0, dx_per_frame=0.0, seed=4)

    raw_tangents = []
    smoothed_tangents = []
    for frame_id, frame in enumerate(frames):
        cx, cy, r = frame[0]
        raw_tangents.append(cx - r)
        result = tracker.update(_dets(frame, frame_id), frame_id)
        smoothed_tangents.append(result.tracks[0].tangent_x)

    raw_var = statistics.pvariance(raw_tangents)
    # Skip the initial transient (EMA hasn't converged in the first few
    # updates) so the comparison reflects steady-state smoothing.
    smoothed_var = statistics.pvariance(smoothed_tangents[10:])
    assert smoothed_var * 4.0 <= raw_var, (
        f"expected >=4x variance reduction: raw_var={raw_var:.3f} smoothed_var={smoothed_var:.3f}"
    )


# ---------------------------------------------------------------------------
# P5 — miss_then_reacquire
# ---------------------------------------------------------------------------


def test_P5_miss_then_reacquire() -> None:
    tracker = PieceTracker(_cfg())

    result0 = tracker.update(
        [RawDetection(center_x=600.0, center_y=200.0, radius=30.0, source_frame_id=0, det_index=0)], 0,
    )
    piece_id = result0.tracks[0].piece_id
    result0.tracks[0].row_id = 42  # simulate RowTracker having assigned a row

    for frame_id in range(1, 4):
        result = tracker.update([], frame_id)
        assert result.missed == [piece_id]
        assert result.dropped == []
        track = next(t for t in result.tracks if t.piece_id == piece_id)
        assert track.row_id == 42, "a miss must never clear row_id"

    result_back = tracker.update(
        [RawDetection(center_x=603.0, center_y=201.0, radius=30.5, source_frame_id=4, det_index=0)], 4,
    )
    assert result_back.created == [], "the same piece_id must be reused, not recreated"
    assert [m[0] for m in result_back.matched] == [piece_id]
    track_back = next(t for t in result_back.tracks if t.piece_id == piece_id)
    assert track_back.row_id == 42
