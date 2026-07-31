"""Tests for ``viscontrol.core.transfer_orchestrator`` (TransferOrchestrator)
and its collaborators (RowAssociator, the pure transfer_events data model).

No Qt. No camera. No PLC. No real clock (FakeClock, manually advanced). No
sleep(). Everything here can run in plain ``pytest`` with no fixtures beyond
what this file defines.
"""

from __future__ import annotations

import random

from viscontrol.core.config import AppConfig
from viscontrol.core.transfer_events import ClusterObservation, EventType, RowState
from viscontrol.core.transfer_orchestrator import TransferOrchestrator


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class FakeClock:
    """Manually-advanced monotonic clock — no wall-clock time anywhere."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeStopSink:
    """Records every TransferSnapshot handed to it, in order."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, snapshot) -> None:  # noqa: ANN001
        self.calls.append(snapshot)


class RecordingLogger:
    """Captures every structured log line the orchestrator emits, so tests
    can assert on prefixes (e.g. "STOP-DUPLICATE-BLOCKED",
    "INVARIANT-VIOLATION") without depending on the real loguru sink."""

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


def make_orchestrator():
    """Fresh orchestrator + its FakeClock/FakeStopSink/RecordingLogger."""
    cfg = AppConfig().transfer_orchestrator
    clock = FakeClock()
    sink = FakeStopSink()
    rec_logger = RecordingLogger()
    orch = TransferOrchestrator(cfg=cfg, stop_command_sink=sink, now_fn=clock, logger=rec_logger)
    return orch, clock, sink, rec_logger, cfg


def obs(front: float, back: float, *, piece_count: int = 1, center_y: float = 100.0, idx: int = 0) -> ClusterObservation:
    return ClusterObservation(
        front_tangent=front, back_tangent=back, piece_count=piece_count,
        center_y=center_y, temp_cluster_index=idx,
    )


def feed(orch: TransferOrchestrator, clock: FakeClock, frame_id: int, observations: list, *, dt: float = 0.05) -> None:
    evt = orch.make_event(EventType.OBSERVATION, source_frame_id=frame_id, observations=tuple(observations))
    orch.submit(evt)
    orch.drain()
    clock.advance(dt)


def create_and_arm_row(
    orch: TransferOrchestrator, clock: FakeClock, cfg, transfer_x: float, *,
    start_front: float, frame_id: int, step: float = 100.0, width: float = 50.0, idx: int = 0,
) -> tuple[int, float]:
    """Feed enough consecutive frames to create a row and arm it
    (DETECTED -> ACTIVE), without crossing transfer_x. Returns
    (next_frame_id, next_front_to_feed_for_a_crossing_frame)."""
    front = start_front
    for _ in range(cfg.row_arm_min_frames + 1):
        feed(orch, clock, frame_id, [obs(front, front + width, idx=idx)])
        frame_id += 1
        front -= step
    return frame_id, front


def row_by_track(orch: TransferOrchestrator, track_id: int):
    for row in orch.snapshot().rows:
        if row.track_id == track_id:
            return row
    return None


# ---------------------------------------------------------------------------
# Test 1 — sequential_three_rows
# ---------------------------------------------------------------------------


def _run_one_row_to_transferred(orch, clock, cfg, transfer_x, *, start_front, frame_id, idx):
    """Drive a single row through its full lifecycle (create -> ACTIVE ->
    STOP_REQUESTED -> STOPPED -> TRANSFER_PENDING -> TRANSFERRING ->
    TRANSFERRED). Returns (row_id, track_id, display_number, next_frame_id)."""
    frame_id, crossing_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=start_front, frame_id=frame_id, idx=idx,
    )
    feed(orch, clock, frame_id, [obs(crossing_front, crossing_front + 50, idx=idx)])
    frame_id += 1

    # Locate the row we just created: it's the one that just crossed.
    candidates = [r for r in orch.snapshot().rows if r.state == RowState.STOP_REQUESTED]
    assert len(candidates) == 1
    row = candidates[0]

    ack_evt = orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=frame_id)
    orch.submit(ack_evt)
    orch.drain()
    clock.advance(0.05)

    cloth_evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    orch.submit(cloth_evt)
    orch.drain()
    clock.advance(0.05)
    frame_id += 1

    exit_front = transfer_x - cfg.row_exit_margin_px - 10
    feed(orch, clock, frame_id, [obs(exit_front, exit_front + 50, idx=idx)])
    frame_id += 1

    final_row = row_by_track(orch, row.track_id)
    assert final_row.state == RowState.TRANSFERRED
    return row.row_id, row.track_id, row.display_number, frame_id


def test_sequential_three_rows() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id = 1
    row_ids = []
    display_numbers = []
    for i in range(3):
        row_id, track_id, disp, frame_id = _run_one_row_to_transferred(
            orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=frame_id, idx=i,
        )
        row_ids.append(row_id)
        display_numbers.append(disp)

    assert len(set(row_ids)) == 3, "expected 3 distinct row_ids"
    assert len(sink.calls) == 3, "sink must be called exactly once per row"
    assert len({s.row_id for s in sink.calls}) == 3, "each snapshot must have a distinct row_id"
    snap = orch.snapshot()
    assert all(r.state == RowState.TRANSFERRED for r in snap.rows)
    assert display_numbers == [1, 2, 3]
    assert not rec_logger.has("INVARIANT-VIOLATION")


# ---------------------------------------------------------------------------
# Test 2 — duplicate_crossing_blocked
# ---------------------------------------------------------------------------


def test_duplicate_crossing_blocked() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id, crossing_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1,
    )
    crossing_evt = orch.make_event(
        EventType.OBSERVATION, source_frame_id=frame_id,
        observations=(obs(crossing_front, crossing_front + 50),),
    )
    orch.submit(crossing_evt)
    orch.drain()
    assert len(sink.calls) == 1

    row = next(r for r in orch.snapshot().rows if r.state == RowState.STOP_REQUESTED)

    # Directly re-invoke the idempotency guard a second time (white-box):
    # the normal frame-processing path structurally cannot re-trigger
    # _request_stop for an already-STOP_REQUESTED row (only ACTIVE rows are
    # tripwire-evaluated), so this exercises the safety-net guard itself —
    # exactly the scenario STOP-DUPLICATE-BLOCKED exists to catch.
    internal_row = orch._rows[row.row_id]  # noqa: SLF001
    orch._request_stop(internal_row, crossing_evt)  # noqa: SLF001

    assert len(sink.calls) == 1, "sink must not be called a second time for the same row"
    assert rec_logger.has("STOP-DUPLICATE-BLOCKED")


# ---------------------------------------------------------------------------
# Test 3 — out_of_order_plc_callback
# ---------------------------------------------------------------------------


def test_out_of_order_plc_callback() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id, _next_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1,
    )
    before = {(r.row_id, r.track_id, r.state) for r in orch.snapshot().rows}
    assert before == {(1, 1, RowState.ACTIVE)}

    # Deliver PLC_CLOTH_START before the expected PLC_STOP_ACK.
    evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    orch.submit(evt)
    orch.drain()

    assert rec_logger.has("no_row_awaiting_transfer")
    assert len(sink.calls) == 0
    after = {(r.row_id, r.track_id, r.state) for r in orch.snapshot().rows}
    assert after == before, "no row_id/track_id/state may change on a rejected out-of-order event"


# ---------------------------------------------------------------------------
# Test 4 — old_cycle_event_rejected
# ---------------------------------------------------------------------------


def test_old_cycle_event_rejected() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    orch.set_transfer_x(1000.0)

    stale_evt = orch.make_event(
        EventType.OBSERVATION, source_frame_id=1,
        observations=(obs(1400.0, 1450.0),),
    )
    rows_before = orch.snapshot().rows

    orch.start_cycle(10)  # bumps cycle_id — stale_evt's cycle_id is now old

    orch.submit(stale_evt)
    orch.drain()

    assert rec_logger.has("OLD-CYCLE-EVENT-BLOCKED")
    assert orch.snapshot().rows == rows_before == ()
    assert len(sink.calls) == 0


# ---------------------------------------------------------------------------
# Test 5 — display_removal_does_not_change_identity
# ---------------------------------------------------------------------------


def test_display_removal_does_not_change_identity() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    # Three well-separated rows, spaced beyond row_new_min_gap_px so each
    # becomes its own row.
    feed(orch, clock, 1, [obs(1400.0, 1450.0, idx=0), obs(1900.0, 1950.0, idx=1), obs(2400.0, 2450.0, idx=2)])

    snap1 = orch.snapshot()
    assert len(snap1.rows) == 3

    # Simulate a UI list mutation: build a NEW tuple/list with row 1 removed.
    filtered = snap1.rows[1:]
    assert len(filtered) == 2

    snap2 = orch.snapshot()
    assert len(snap2.rows) == 3, "the orchestrator's own row set must be unaffected by a UI-side mutation"

    # Rows 2 and 3 (by creation order == display_number 2 and 3) must have
    # identical row_id/track_id in both snapshots.
    for disp in (2, 3):
        r1 = next(r for r in snap1.rows if r.display_number == disp)
        r2 = next(r for r in snap2.rows if r.display_number == disp)
        assert r1.row_id == r2.row_id
        assert r1.track_id == r2.track_id

    # Mutating a returned copy must never affect internal state.
    snap1.rows[0].state = RowState.TRANSFERRED  # type: ignore[misc]
    snap3 = orch.snapshot()
    assert snap3.rows[0].state == RowState.DETECTED


# ---------------------------------------------------------------------------
# Test 6 — transfer_snapshot_is_row_scoped
# ---------------------------------------------------------------------------


def test_transfer_snapshot_is_row_scoped() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    # Decoy rows far away (never re-fed, so they just accumulate harmless
    # missed_frames — well under row_max_missed_frames for this test's
    # short frame count) so they can never be confused with the row we
    # actually drive, whose descent (1400 -> 1000) stays nowhere near them.
    feed(orch, clock, 1, [obs(5000.0, 5050.0, idx=0), obs(6000.0, 6050.0, idx=2)])
    frame_id, crossing_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=2, idx=1,
    )
    feed(orch, clock, frame_id, [obs(crossing_front, crossing_front + 50, idx=1)])
    frame_id += 1

    row2 = next(r for r in orch.snapshot().rows if r.state == RowState.STOP_REQUESTED)
    assert len(sink.calls) == 1
    snap_at_stop = sink.calls[0]
    assert snap_at_stop.row_id == row2.row_id
    assert f"R{row2.row_id}" in snap_at_stop.stop_event_id
    other_row_ids = [r.row_id for r in orch.snapshot().rows if r.row_id != row2.row_id]
    for other_id in other_row_ids:
        assert f"R{other_id}:" not in snap_at_stop.stop_event_id

    ack_evt = orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=frame_id)
    orch.submit(ack_evt)
    orch.drain()
    cloth_evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    orch.submit(cloth_evt)
    orch.drain()

    row2_final = row_by_track(orch, row2.track_id)
    assert row2_final.state == RowState.TRANSFERRING
    assert row2_final.row_id == row2.row_id


# ---------------------------------------------------------------------------
# Test 7 — cluster_split_merge_preserves_identity
# ---------------------------------------------------------------------------


def test_cluster_split_merge_preserves_identity() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id, _next = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1,
    )
    row_before = next(iter(orch.snapshot().rows))
    assert row_before.state == RowState.ACTIVE

    # ONE observation (whole row visible as a single cluster).
    feed(orch, clock, frame_id, [obs(1090.0, 1140.0, idx=0)])
    frame_id += 1

    # TWO observations, 87px apart — the same physical row momentarily
    # split into two columns.
    feed(orch, clock, frame_id, [obs(1080.0, 1100.0, idx=0), obs(1167.0, 1190.0, idx=1)])
    frame_id += 1

    # Back to ONE observation.
    feed(orch, clock, frame_id, [obs(1070.0, 1140.0, idx=0)])
    frame_id += 1

    rows = orch.snapshot().rows
    assert len(rows) == 1, "no new row may be created by the split/merge"
    row_after = rows[0]
    assert row_after.row_id == row_before.row_id
    assert row_after.track_id == row_before.track_id
    assert row_after.display_number == row_before.display_number
    assert not rec_logger.has("INVARIANT-VIOLATION")


# ---------------------------------------------------------------------------
# Test 8 — deterministic_under_shuffled_ordering
# ---------------------------------------------------------------------------


def _run_shuffled_lifecycle(seed: int):
    """One full single-row lifecycle. At two points, a PLC event and the
    immediately-following (order-independent — see comments) observation
    event are submitted in a shuffled order within a single drain() call.
    Returns (final_row_table, sink_call_count) for cross-run comparison.
    """
    rng = random.Random(seed)
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id, crossing_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1,
    )
    feed(orch, clock, frame_id, [obs(crossing_front, crossing_front + 50)])
    frame_id += 1  # row is now STOP_REQUESTED

    # Pair 1: PLC_STOP_ACK + the next observation frame. Order-independent:
    # the ack only ever moves STOP_REQUESTED -> TRANSFER_PENDING (never
    # TRANSFERRING), so the observation's exit-margin check (which only
    # fires for TRANSFERRING rows) never triggers regardless of order.
    ack_evt = orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=frame_id)
    obs_evt_1 = orch.make_event(
        EventType.OBSERVATION, source_frame_id=frame_id,
        observations=(obs(990.0, 1040.0),),
    )
    pair1 = [ack_evt, obs_evt_1]
    rng.shuffle(pair1)
    for e in pair1:
        orch.submit(e)
    orch.drain()
    clock.advance(0.05)
    frame_id += 1

    # Pair 2: PLC_CLOTH_START + the next observation frame, chosen so its
    # front_tangent (950) stays above the exit-margin threshold (920)
    # regardless of whether cloth-start has already flipped the row to
    # TRANSFERRING by the time the observation applies its match — so the
    # exit transition never fires in this pair either way.
    cloth_evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    obs_evt_2 = orch.make_event(
        EventType.OBSERVATION, source_frame_id=frame_id,
        observations=(obs(950.0, 1000.0),),
    )
    pair2 = [cloth_evt, obs_evt_2]
    rng.shuffle(pair2)
    for e in pair2:
        orch.submit(e)
    orch.drain()
    clock.advance(0.05)
    frame_id += 1

    # Final, unshuffled frame: genuinely exits (900 < transfer_x - margin=920).
    feed(orch, clock, frame_id, [obs(900.0, 950.0)])

    table = tuple(
        (r.row_id, r.track_id, r.state.value, r.stop_event_id)
        for r in orch.snapshot().rows
    )
    assert not rec_logger.has("INVARIANT-VIOLATION")
    return table, len(sink.calls)


def test_deterministic_under_shuffled_ordering() -> None:
    results = [_run_shuffled_lifecycle(seed) for seed in range(100)]
    first = results[0]
    assert all(r == first for r in results), "all 100 shuffled runs must produce identical final state"
    assert first[0][0][2] == RowState.TRANSFERRED.value
    assert first[1] == 1


# ---------------------------------------------------------------------------
# Test 9 — detection_miss_never_transfers
# ---------------------------------------------------------------------------


def test_detection_miss_never_transfers() -> None:
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id, crossing_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1,
    )
    feed(orch, clock, frame_id, [obs(crossing_front, crossing_front + 50)])
    frame_id += 1
    assert len(sink.calls) == 1
    row = next(r for r in orch.snapshot().rows if r.state == RowState.STOP_REQUESTED)

    # Total detection loss: 50 frames with zero observations. Advance the
    # clock only a little each frame (well under stop_ack_timeout_s) so the
    # row is not forced to ABANDONED by the timeout either — this isolates
    # the "detection miss alone" case the CRITICAL RULE protects.
    for _ in range(50):
        feed(orch, clock, frame_id, [], dt=0.01)
        frame_id += 1

    final = row_by_track(orch, row.track_id)
    assert final.state != RowState.TRANSFERRED
    assert final.state in (RowState.STOP_REQUESTED, RowState.ABANDONED)
    assert len(sink.calls) == 1, "the sink must never be called a second time"


# ---------------------------------------------------------------------------
# Test 10 — invariant_never_violated
# ---------------------------------------------------------------------------


def test_invariant_never_violated() -> None:
    """A composite run touching the trickier interactions (duplicate
    crossing attempt, out-of-order PLC callback, shuffled ordering, total
    detection loss) — asserts _assert_invariants() never raised and no
    INVARIANT-VIOLATION was logged, across all of it."""
    orch, clock, sink, rec_logger, cfg = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    # Two independent rows, driven concurrently.
    frame_id, front1 = create_and_arm_row(orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1, idx=0)
    frame_id2, front2 = create_and_arm_row(orch, clock, cfg, transfer_x, start_front=1900.0, frame_id=frame_id, idx=1)
    frame_id = frame_id2

    # Row 1 crosses.
    feed(orch, clock, frame_id, [obs(front1, front1 + 50, idx=0), obs(front2, front2 + 50, idx=1)])
    frame_id += 1

    # Duplicate-crossing safety net, exercised directly (white-box, as in
    # test_duplicate_crossing_blocked).
    row1 = next(r for r in orch.snapshot().rows if r.state == RowState.STOP_REQUESTED)
    internal_row1 = orch._rows[row1.row_id]  # noqa: SLF001
    dummy_evt = orch.make_event(EventType.OBSERVATION, source_frame_id=frame_id)
    orch._request_stop(internal_row1, dummy_evt)  # noqa: SLF001

    # Out-of-order PLC callback for row 1's transfer (no ack yet).
    evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    orch.submit(evt)
    orch.drain()

    # Now legitimately ack + start row 1's transfer while row 2 continues.
    ack_evt = orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=frame_id)
    orch.submit(ack_evt)
    orch.drain()
    cloth_evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    orch.submit(cloth_evt)
    orch.drain()
    frame_id += 1

    exit_front = transfer_x - cfg.row_exit_margin_px - 10
    feed(orch, clock, frame_id, [obs(exit_front, exit_front + 50, idx=0), obs(front2 - 50, front2, idx=1)])
    frame_id += 1

    # Total detection loss for a few frames.
    for _ in range(10):
        feed(orch, clock, frame_id, [], dt=0.01)
        frame_id += 1

    assert not rec_logger.has("INVARIANT-VIOLATION")
    # Also re-run the dedicated shuffle scenario's invariant check.
    for seed in range(20):
        _run_shuffled_lifecycle(seed)
