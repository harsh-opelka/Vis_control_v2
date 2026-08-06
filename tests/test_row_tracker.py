"""Tests for viscontrol.core.row_tracker (RowTracker).

No Qt, no cv2, no camera, no PLC, no real clock, no sleep().

Two tests (C, D) exercise a genuinely TEMPORARY split/merge proposal (a
couple of frames, well under the evidence-frame thresholds) that reverts
before ever being accepted. RowTracker's own log vocabulary only emits the
literal *-REJECTED prefix when the locked/near-lock guard fires; a
transient proposal that simply never accumulates enough evidence logs
*-PROPOSED (and, on reverting, a DEBUG evidence-reset line) instead — so
those two tests assert on the invariant that actually matters (no
*-ACCEPTED, membership/identity untouched) rather than a specific log
prefix. Test G covers the locked-guard path explicitly.
"""

from __future__ import annotations

import random

from viscontrol.core.config import AppConfig
from viscontrol.core.piece_tracker import PieceTracker, RawDetection
from viscontrol.core.row_tracker import RowGroupState, RowTracker
from tests.synthetic_rows import RecordingLogger, make_frames, make_track


def _cfg(**overrides):
    cfg = AppConfig().row_tracker
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg


_FAR_UPSTREAM = -1_000_000.0  # transfer_x far away so nothing locks/transfers mid-test


# ---------------------------------------------------------------------------
# A — stationary_close_rows
# ---------------------------------------------------------------------------


def test_A_stationary_close_rows() -> None:
    piece_tracker = PieceTracker(AppConfig().piece_tracker)
    row_tracker = RowTracker(_cfg())

    # Each row's 2 members sit at DIFFERENT Y (columns) — two pieces can
    # never share a Y lane at all (that's the physical non-overlap
    # constraint, regardless of which row they end up in), so a valid
    # multi-member row always has members spread across distinct Y
    # positions. Within-row gap 60px (<= row_x_tolerance_px=110, always
    # merges); between-row gap 150px (> row_x_tolerance_px=110, always
    # separates by X alone — Y-exclusivity isn't what's under test here,
    # that's B/H's job). Center_x = tangent_x + radius(30).
    rows_layout = [
        [(530.0, 50.0, 30.0), (590.0, 200.0, 30.0)],     # row 1: tangent 500, 560
        [(740.0, 50.0, 30.0), (800.0, 200.0, 30.0)],     # row 2: tangent 710, 770
        [(950.0, 50.0, 30.0), (1010.0, 200.0, 30.0)],    # row 3: tangent 920, 980
    ]
    frames = make_frames(rows_layout, n_frames=40, jitter_px=15.0, dx_per_frame=0.0, seed=5)

    row_id_sets: list[frozenset] = []
    for frame_id, frame in enumerate(frames):
        dets = [
            RawDetection(center_x=cx, center_y=cy, radius=r, source_frame_id=frame_id, det_index=i)
            for i, (cx, cy, r) in enumerate(frame)
        ]
        piece_result = piece_tracker.update(dets, frame_id)
        stable = row_tracker.update(piece_result.tracks, frame_id, _FAR_UPSTREAM)
        if frame_id >= 15:
            row_id_sets.append(frozenset(o.row_id for o in stable))

    final = row_id_sets[-1]
    assert len(final) == 3, f"expected 3 stable rows, got {sorted(final)}"
    assert all(s == final for s in row_id_sets[-10:]), "row_ids must not change once stable"


# ---------------------------------------------------------------------------
# B — threshold_ambiguity
# ---------------------------------------------------------------------------


def test_B_threshold_ambiguity() -> None:
    # Shape of the recorded geometry (see BACKGROUND): a within-row gap and
    # a between-row gap that overlap in range, with the between-row PAIR
    # sharing center_y lanes (row 1's nearest-to-row-2 piece reuses row 2's
    # Y positions). The within-row gap here (40px) is tightened from the
    # recorded ~63.5px so the SAME 3-row answer holds at all three tested
    # tolerances, including the tightest (50px) — RowTracker's own grouping
    # sweep requires x_gap <= tolerance to ever join two pieces, so a
    # within-row gap that a tolerance value structurally can't bridge could
    # never be fixed by Y-exclusivity (Y only ever SEPARATES, never joins).
    # What this test proves is the part Y-exclusivity actually governs: at
    # the LOOSER tolerances (85, 150) X alone would wrongly merge row 1 and
    # row 2 (between-row gap 74px < tolerance) were it not for the shared
    # Y lanes forcing them apart.
    for tol in (50.0, 85.0, 150.0):
        cfg = _cfg(row_x_tolerance_px=tol)
        tracker = RowTracker(cfg)
        tracks = [
            make_track(1, 460.0, 350.0, 30.0),   # row 1, farthest column
            make_track(2, 500.0, 200.0, 30.0),   # row 1, middle column
            make_track(3, 540.0, 50.0, 30.0),    # row 1, nearest column — shares Y with piece 4
            make_track(4, 614.0, 50.0, 30.0),    # row 2 — between-row gap from piece 3: 74px
            make_track(5, 624.0, 200.0, 30.0),   # row 2 — shares Y with piece 2
            make_track(6, 820.0, 500.0, 30.0),   # row 3 — far in both X and Y, always separate
            make_track(7, 846.0, 650.0, 30.0),
        ]
        stable = []
        for frame_id in range(8):
            stable = tracker.update(tracks, frame_id, _FAR_UPSTREAM)
        assert len(stable) == 3, (
            f"tol={tol} produced {len(stable)} rows: "
            f"{[(o.row_id, o.front_tangent, o.back_tangent) for o in stable]}"
        )


# ---------------------------------------------------------------------------
# C — temporary_split
# ---------------------------------------------------------------------------


def test_C_temporary_split() -> None:
    rec_logger = RecordingLogger()
    tracker = RowTracker(_cfg(), logger=rec_logger)

    t1 = make_track(1, 460.0, 50.0, 30.0)
    t2 = make_track(2, 500.0, 200.0, 30.0)  # gap 40, distinct column — same row

    stable = []
    for frame_id in range(6):
        stable = tracker.update([t1, t2], frame_id, _FAR_UPSTREAM)
    assert len(stable) == 1
    row_before = stable[0]
    assert row_before.piece_count == 2
    version_before = row_before.membership_version
    members_before = frozenset(tracker._rows[row_before.row_id].member_piece_ids)  # noqa: SLF001

    # Temporarily appears as two separate groups (push piece 2 far away in
    # X) for exactly 2 frames.
    t2_split = make_track(2, 500.0 + 600.0, 400.0, 30.0)
    tracker.update([t1, t2_split], 6, _FAR_UPSTREAM)
    tracker.update([t1, t2_split], 7, _FAR_UPSTREAM)

    # Revert.
    stable_after = []
    for frame_id in range(8, 14):
        stable_after = tracker.update([t1, t2], frame_id, _FAR_UPSTREAM)

    assert len(stable_after) == 1
    row_after = stable_after[0]
    assert row_after.row_id == row_before.row_id
    assert row_after.membership_version == version_before
    assert frozenset(tracker._rows[row_after.row_id].member_piece_ids) == members_before  # noqa: SLF001
    assert not rec_logger.has("ROW-SPLIT-ACCEPTED")


# ---------------------------------------------------------------------------
# D — temporary_merge
# ---------------------------------------------------------------------------


def test_D_temporary_merge() -> None:
    rec_logger = RecordingLogger()
    tracker = RowTracker(_cfg(), logger=rec_logger)

    t1 = make_track(1, 400.0, 100.0, 30.0)
    t2 = make_track(2, 900.0, 300.0, 30.0)  # gap 500, distinct column — always its own row

    for frame_id in range(6):
        stable = tracker.update([t1, t2], frame_id, _FAR_UPSTREAM)
    assert len(stable) == 2
    row_ids_before = {o.row_id for o in stable}
    versions_before = {o.row_id: o.membership_version for o in stable}

    # Temporarily overlaps into a single group for exactly 2 frames (piece
    # 2's column stays distinct from piece 1's, so nothing but the X gap
    # itself decides whether they group).
    t2_close = make_track(2, 450.0, 300.0, 30.0)  # gap from t1 now 50px
    tracker.update([t1, t2_close], 6, _FAR_UPSTREAM)
    tracker.update([t1, t2_close], 7, _FAR_UPSTREAM)

    # Revert.
    for frame_id in range(8, 16):
        stable_after = tracker.update([t1, t2], frame_id, _FAR_UPSTREAM)

    assert {o.row_id for o in stable_after} == row_ids_before, "both row_ids must survive"
    for o in stable_after:
        assert o.membership_version == versions_before[o.row_id]
    assert not rec_logger.has("ROW-MERGE-ACCEPTED")


# ---------------------------------------------------------------------------
# E — missed_detection
# ---------------------------------------------------------------------------


def test_E_missed_detection() -> None:
    tracker = RowTracker(_cfg())

    t1 = make_track(1, 460.0, 50.0, 30.0)
    t2 = make_track(2, 500.0, 200.0, 30.0)  # distinct column — same row

    for frame_id in range(6):
        stable = tracker.update([t1, t2], frame_id, _FAR_UPSTREAM)
    assert len(stable) == 1
    row_id = stable[0].row_id
    version_before = stable[0].membership_version

    # Piece 2 is absent from the tracks list for 3 consecutive frames (as if
    # PieceTracker briefly lost it) — well under row_remove_evidence_frames (5).
    for frame_id in range(6, 9):
        stable = tracker.update([t1], frame_id, _FAR_UPSTREAM)
        assert len(stable) == 1
        assert stable[0].row_id == row_id
        assert stable[0].state == RowGroupState.CONFIRMED
        assert stable[0].membership_version == version_before

    assert tracker._rows[row_id].member_piece_ids == {1, 2}  # noqa: SLF001


# ---------------------------------------------------------------------------
# F — detection_order_shuffle
# ---------------------------------------------------------------------------


def _run_full_pipeline_shuffled(frames: list, shuffle_seed: int) -> list:
    piece_tracker = PieceTracker(AppConfig().piece_tracker)
    row_tracker = RowTracker(_cfg())
    rng = random.Random(shuffle_seed)
    history = []
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
        piece_result = piece_tracker.update(dets, frame_id)
        stable = row_tracker.update(piece_result.tracks, frame_id, _FAR_UPSTREAM)
        history.append((
            frozenset(t.piece_id for t in piece_result.tracks),
            frozenset(o.row_id for o in stable),
        ))
    return history


def test_F_detection_order_shuffle() -> None:
    rows_layout = [
        [(530.0, 50.0, 30.0), (590.0, 200.0, 30.0)],    # tangent 500, 560
        [(740.0, 50.0, 30.0), (800.0, 200.0, 30.0)],    # tangent 710, 770 (gap 150 > tolerance)
    ]
    frames = make_frames(rows_layout, n_frames=50, jitter_px=15.0, dx_per_frame=0.0, seed=6)

    baseline = _run_full_pipeline_shuffled(frames, shuffle_seed=999)
    for seed in range(10):
        result = _run_full_pipeline_shuffled(frames, shuffle_seed=seed)
        assert result == baseline, f"shuffle seed={seed} diverged from the baseline run"


# ---------------------------------------------------------------------------
# G — locked_row_immutable
# ---------------------------------------------------------------------------


def test_G_locked_row_immutable() -> None:
    tracker = RowTracker(_cfg())

    t1 = make_track(1, 460.0, 50.0, 30.0)
    t2 = make_track(2, 500.0, 200.0, 30.0)  # distinct column — same row
    transfer_x = 460.0 - 10.0  # comfortably within row_lock_margin_px (150) from frame 0

    stable = []
    for frame_id in range(6):
        stable = tracker.update([t1, t2], frame_id, transfer_x)
    assert len(stable) == 1
    row = stable[0]
    assert row.state == RowGroupState.LOCKED_FOR_TRANSFER
    snapshot_before = row.locked_snapshot
    assert snapshot_before is not None
    members_before = frozenset(tracker._rows[row.row_id].member_piece_ids)  # noqa: SLF001
    version_before = row.membership_version

    # A frame whose proposed grouping would both split the locked row's
    # existing members AND add a new nearby piece. t3_new sits in a
    # DIFFERENT Y lane from the row (so Y-exclusivity doesn't block it from
    # proposing to join) and close to t1 in tangent_x — it ends up grouped
    # WITH t1 in the proposal, splitting the locked row's original {1, 2}
    # into two proposed groups ({2} alone, {1, 3} together).
    t2_moved = make_track(2, 900.0, 500.0, 30.0)  # far in X and Y — proposes a split
    t3_new = make_track(3, 465.0, 300.0, 30.0)    # would otherwise join/merge with t1

    for frame_id in range(6, 20):
        stable = tracker.update([t1, t2_moved, t3_new], frame_id, transfer_x)

    row_after = next(o for o in stable if o.row_id == row.row_id)
    assert row_after.state == RowGroupState.LOCKED_FOR_TRANSFER
    assert row_after.membership_version == version_before
    assert frozenset(tracker._rows[row.row_id].member_piece_ids) == members_before  # noqa: SLF001
    assert row_after.locked_snapshot is snapshot_before, "LockedRowSnapshot must never be rebuilt once locked"


# ---------------------------------------------------------------------------
# H — y_exclusion_separates
# ---------------------------------------------------------------------------


def test_H_y_exclusion_separates() -> None:
    # Same X geometry (60px apart in tangent_x) in both cases; only Y changes.
    cfg = _cfg()

    tracker_same_lane = RowTracker(cfg)
    t1 = make_track(1, 500.0, 100.0, 30.0)
    t2 = make_track(2, 560.0, 100.0, 30.0)  # shares the Y lane -> must SEPARATE
    stable = []
    for frame_id in range(6):
        stable = tracker_same_lane.update([t1, t2], frame_id, _FAR_UPSTREAM)
    assert len({o.row_id for o in stable}) == 2, "pieces sharing a Y lane must land in different rows"

    tracker_far_lane = RowTracker(cfg)
    t1b = make_track(1, 500.0, 100.0, 30.0)
    t2b = make_track(2, 560.0, 300.0, 30.0)  # 200px away in Y -> must SHARE a row
    stable2 = []
    for frame_id in range(6):
        stable2 = tracker_far_lane.update([t1b, t2b], frame_id, _FAR_UPSTREAM)
    assert len(stable2) == 1, "pieces far apart in Y must land in the same row"
    assert stable2[0].piece_count == 2
