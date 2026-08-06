"""Integration replay tests: PieceTracker -> RowTracker -> TransferOrchestrator,
end to end.

No Qt, no cv2, no camera, no PLC, no real clock, no sleep(). Reuses
FakeClock/FakeStopSink/RecordingLogger from test_transfer_orchestrator.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from viscontrol.core.config import AppConfig
from viscontrol.core.detection_recorder import load_recording
from viscontrol.core.piece_tracker import PieceTracker, RawDetection
from viscontrol.core.row_tracker import RowTracker
from viscontrol.core.transfer_events import EventType, RowState
from viscontrol.core.transfer_orchestrator import TransferOrchestrator
from tests.synthetic_rows import make_frames
from tests.test_transfer_orchestrator import FakeClock, FakeStopSink, RecordingLogger

_THREE_ROWS_LAYOUT = [
    [(1500.0, 50.0, 30.0), (1560.0, 200.0, 30.0)],
    [(2100.0, 50.0, 30.0), (2160.0, 200.0, 30.0)],
    [(2700.0, 50.0, 30.0), (2760.0, 200.0, 30.0)],
]

_WIDELY_SEPARATED_LAYOUT = [
    [(1500.0, 50.0, 30.0), (1560.0, 200.0, 30.0)],
    [(2500.0, 50.0, 30.0), (2560.0, 200.0, 30.0)],
    [(3500.0, 50.0, 30.0), (3560.0, 200.0, 30.0)],
]


def _run_pipeline(rows_layout, *, n_frames, jitter_px, dx_per_frame, transfer_x, seed):
    """Drives PieceTracker -> RowTracker -> TransferOrchestrator over a
    synthetic recording, auto-acking every stop request (as a real,
    responsive PLC would) so rows complete their full lifecycle.

    Returns (orch, sink, rec_logger, piece_tracker, row_tracker).
    """
    clock = FakeClock()
    sink = FakeStopSink()
    rec_logger = RecordingLogger()
    orch = TransferOrchestrator(
        cfg=AppConfig().transfer_orchestrator, stop_command_sink=sink,
        now_fn=clock, logger=rec_logger,
    )
    piece_tracker = PieceTracker(AppConfig().piece_tracker)
    row_tracker = RowTracker(AppConfig().row_tracker)
    orch.start_cycle(0)
    orch.set_transfer_x(transfer_x)

    frames = make_frames(
        rows_layout, n_frames=n_frames, jitter_px=jitter_px, dx_per_frame=dx_per_frame, seed=seed,
    )

    for frame_id, frame in enumerate(frames):
        dets = [
            RawDetection(center_x=cx, center_y=cy, radius=r, source_frame_id=frame_id, det_index=i)
            for i, (cx, cy, r) in enumerate(frame)
        ]
        piece_result = piece_tracker.update(dets, frame_id)
        stable_rows = row_tracker.update(piece_result.tracks, frame_id, transfer_x)
        evt = orch.make_event(
            EventType.OBSERVATION, source_frame_id=frame_id, observations=tuple(stable_rows),
        )
        orch.submit(evt)
        orch.drain()
        clock.advance(0.1)

        if any(r.state == RowState.STOP_REQUESTED for r in orch.snapshot().rows):
            orch.submit(orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=frame_id))
            orch.drain()
            clock.advance(0.05)
            orch.submit(orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id))
            orch.drain()
            clock.advance(0.05)

    return orch, sink, rec_logger, piece_tracker, row_tracker


# ---------------------------------------------------------------------------
# R1 — full_pipeline_three_rows
# ---------------------------------------------------------------------------


def test_R1_full_pipeline_three_rows() -> None:
    orch, sink, rec_logger, _pt, _rt = _run_pipeline(
        _THREE_ROWS_LAYOUT, n_frames=250, jitter_px=10.0, dx_per_frame=-15.0,
        transfer_x=0.0, seed=1,
    )

    assert len(sink.calls) == 3, "sink must be called exactly once per row"
    fired_row_ids = {s.row_id for s in sink.calls}
    assert len(fired_row_ids) == 3, "each sink call must carry a distinct row_id"

    snap = orch.snapshot()
    assert len({r.row_id for r in snap.rows}) == 3
    assert sum(1 for r in snap.rows if r.state == RowState.TRANSFERRED) == 3

    locked_snapshots = [orch._rows[rid].row_locked_snapshot for rid in fired_row_ids]  # noqa: SLF001
    assert all(s is not None for s in locked_snapshots), "every fired row must carry a LockedRowSnapshot"
    assert len(locked_snapshots) == 3

    member_sets = [set(s.member_piece_ids) for s in locked_snapshots]
    for i in range(len(member_sets)):
        for j in range(i + 1, len(member_sets)):
            assert not (member_sets[i] & member_sets[j]), (
                "no piece may belong to two different LockedRowSnapshots"
            )

    assert not rec_logger.has("INVARIANT-VIOLATION")


# ---------------------------------------------------------------------------
# R2 — replay_determinism
# ---------------------------------------------------------------------------


def test_R2_replay_determinism() -> None:
    baseline = None
    for _ in range(100):
        orch, sink, rec_logger, _pt, _rt = _run_pipeline(
            _THREE_ROWS_LAYOUT, n_frames=250, jitter_px=10.0, dx_per_frame=-15.0,
            transfer_x=0.0, seed=1,
        )
        snap = orch.snapshot()
        stop_event_ids = tuple(
            row.stop_event_id
            for row in sorted(orch._rows.values(), key=lambda r: r.row_id)  # noqa: SLF001
            if row.stop_event_id
        )
        result = (
            tuple(sorted(r.row_id for r in snap.rows)),
            stop_event_ids,
            len(sink.calls),
        )
        if baseline is None:
            baseline = result
        else:
            assert result == baseline, "replay must be fully deterministic run to run"
        assert not rec_logger.has("INVARIANT-VIOLATION")


# ---------------------------------------------------------------------------
# R3 — recorded_log_replay
# ---------------------------------------------------------------------------


def test_R3_recorded_log_replay() -> None:
    path = Path(__file__).parent / "data" / "failing_case.jsonl"
    if not path.exists():
        pytest.skip(
            "tests/data/failing_case.jsonl not present — the operator will add "
            "it after the next run with detection_recorder.enabled=true to "
            "exercise this replay against the real failing case."
        )

    recorded_frames = load_recording(path)
    clock = FakeClock()
    sink = FakeStopSink()
    rec_logger = RecordingLogger()
    orch = TransferOrchestrator(
        cfg=AppConfig().transfer_orchestrator, stop_command_sink=sink,
        now_fn=clock, logger=rec_logger,
    )
    piece_tracker = PieceTracker(AppConfig().piece_tracker)
    row_tracker = RowTracker(AppConfig().row_tracker)
    orch.start_cycle(0)

    for rf in recorded_frames:
        if not rf.is_fresh:
            continue
        orch.set_transfer_x(rf.transfer_x)
        dets = [
            RawDetection(
                center_x=d["center_x"], center_y=d["center_y"], radius=d["radius"],
                source_frame_id=rf.frame_id, det_index=i,
            )
            for i, d in enumerate(rf.detections)
        ]
        piece_result = piece_tracker.update(dets, rf.frame_id)
        stable_rows = row_tracker.update(piece_result.tracks, rf.frame_id, rf.transfer_x)
        evt = orch.make_event(
            EventType.OBSERVATION, source_frame_id=rf.frame_id, observations=tuple(stable_rows),
        )
        orch.submit(evt)
        orch.drain()
        clock.advance(0.1)

        if any(r.state == RowState.STOP_REQUESTED for r in orch.snapshot().rows):
            orch.submit(orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=rf.frame_id))
            orch.drain()
            orch.submit(orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=rf.frame_id))
            orch.drain()

    snap = orch.snapshot()
    assert len(sink.calls) == 3
    assert sum(1 for r in snap.rows if r.state == RowState.TRANSFERRED) == 3


# ---------------------------------------------------------------------------
# R4 — separated_rows_regression
# ---------------------------------------------------------------------------


def test_R4_separated_rows_regression() -> None:
    orch, sink, rec_logger, _pt, _rt = _run_pipeline(
        _WIDELY_SEPARATED_LAYOUT, n_frames=320, jitter_px=10.0, dx_per_frame=-15.0,
        transfer_x=0.0, seed=2,
    )
    assert len(sink.calls) == 3, "no overstopping/understopping for widely-separated rows"
    assert len({s.row_id for s in sink.calls}) == 3
    assert not rec_logger.has("INVARIANT-VIOLATION")
