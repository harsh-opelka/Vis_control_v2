"""Tests for the orchestrator cycle-lifecycle fix: ONE PHYSICAL CLOTH = ONE
ORCHESTRATOR CYCLE. A cloth stops and restarts once per row; those restarts
are mid-cycle events (PLC_STOP_ACK / PLC_CLOTH_START) and must never call
start_cycle() or abandon rows. start_cycle() is reserved for an explicit
session start or the idle-based fallback (see
viscontrol/core/transfer_orchestrator.py: TransferOrchestrator.start_cycle).

No Qt. No camera. No PLC. No real clock (FakeClock, manually advanced). No
sleep(). Reuses FakeClock/FakeStopSink/RecordingLogger and the row-driving
helpers from tests/test_transfer_orchestrator.py.
"""

from __future__ import annotations

from viscontrol.core.config import AppConfig
from viscontrol.core.transfer_events import EventType, RowState
from viscontrol.core.transfer_orchestrator import TransferOrchestrator

from tests.test_transfer_orchestrator import (
    FakeClock,
    FakeStopSink,
    RecordingLogger,
    _run_one_row_to_transferred,
    create_and_arm_row,
    feed,
    obs,
    row_by_track,
)


def make_orchestrator(cycle_idle_reset_ms: int | None = None):
    """Fresh orchestrator + its FakeClock/FakeStopSink/RecordingLogger, plus
    the idle-reset threshold (ms) actually wired into it."""
    full_cfg = AppConfig()
    cfg = full_cfg.transfer_orchestrator
    idle_ms = (
        cycle_idle_reset_ms if cycle_idle_reset_ms is not None
        else full_cfg.detection.cycle_idle_reset_ms
    )
    clock = FakeClock()
    sink = FakeStopSink()
    rec_logger = RecordingLogger()
    orch = TransferOrchestrator(
        cfg=cfg, stop_command_sink=sink, now_fn=clock, logger=rec_logger,
        cycle_idle_reset_ms=idle_ms,
    )
    return orch, clock, sink, rec_logger, cfg, idle_ms


def _cycle_start_lines(rec_logger: RecordingLogger) -> list[str]:
    return [text for _, text in rec_logger.lines if "CYCLE-START" in text]


# ---------------------------------------------------------------------------
# C1 — rising_edge_does_not_restart_cycle
# ---------------------------------------------------------------------------


def test_rising_edge_does_not_restart_cycle() -> None:
    orch, clock, sink, rec_logger, cfg, _idle_ms = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    # Row A: create + arm + cross + ack -> TRANSFER_PENDING.
    frame_id, crossing_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1, idx=0,
    )
    feed(orch, clock, frame_id, [obs(crossing_front, crossing_front + 50, idx=0)])
    frame_id += 1
    row_a = next(r for r in orch.snapshot().rows if r.state == RowState.STOP_REQUESTED)
    ack_evt = orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=frame_id)
    orch.submit(ack_evt)
    orch.drain()
    clock.advance(0.05)
    row_a = row_by_track(orch, row_a.track_id)
    assert row_a.state == RowState.TRANSFER_PENDING

    # Rows B and C: create + arm only (stay ACTIVE), well separated from A
    # and from each other so clustering can never merge/confuse them.
    frame_id, _ = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=2200.0, frame_id=frame_id, idx=1,
    )
    frame_id, _ = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=3000.0, frame_id=frame_id, idx=2,
    )
    active_rows = [r for r in orch.snapshot().rows if r.state == RowState.ACTIVE]
    assert len(active_rows) == 2
    row_b, row_c = active_rows
    b_before = (row_b.row_id, row_b.state)
    c_before = (row_c.row_id, row_c.state)

    cycle_id_before = orch.snapshot().cycle_id

    # PLC rising edge: row A's cloth restart.
    cloth_evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    orch.submit(cloth_evt)
    orch.drain()

    snap = orch.snapshot()
    assert snap.cycle_id == cycle_id_before, "start_cycle must not fire on a rising edge"
    row_a_after = row_by_track(orch, row_a.track_id)
    assert row_a_after.state == RowState.TRANSFERRING
    assert row_a_after.row_id == row_a.row_id

    row_b_after = row_by_track(orch, row_b.track_id)
    row_c_after = row_by_track(orch, row_c.track_id)
    assert (row_b_after.row_id, row_b_after.state) == b_before
    assert (row_c_after.row_id, row_c_after.state) == c_before

    assert snap.rows_abandoned == 0
    assert rec_logger.has("CYCLE-CONTINUE")
    # Only the initial explicit start_cycle(0) may have logged CYCLE-START.
    assert len(_cycle_start_lines(rec_logger)) == 1


# ---------------------------------------------------------------------------
# C2 — one_cloth_three_rows
# ---------------------------------------------------------------------------


def test_one_cloth_three_rows() -> None:
    orch, clock, sink, rec_logger, cfg, _idle_ms = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)
    cycle_id_before = orch.snapshot().cycle_id

    frame_id = 1
    row_ids = []
    for i in range(3):
        row_id, _track_id, _disp, frame_id = _run_one_row_to_transferred(
            orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=frame_id, idx=i,
        )
        row_ids.append(row_id)

    assert len(set(row_ids)) == 3, "expected 3 distinct row_ids"
    assert len(sink.calls) == 3, "exactly 3 stop commands, one per row"
    assert len({s.row_id for s in sink.calls}) == 3, "3 distinct row_ids at the sink"

    snap = orch.snapshot()
    assert all(r.state == RowState.TRANSFERRED for r in snap.rows)
    assert snap.rows_abandoned == 0
    assert snap.cycle_id == cycle_id_before, "one physical cloth = one cycle"
    assert len(_cycle_start_lines(rec_logger)) == 1, "exactly one CYCLE-START for the whole cloth"


# ---------------------------------------------------------------------------
# C3 — idle_timeout_starts_new_cycle
# ---------------------------------------------------------------------------


def test_idle_timeout_starts_new_cycle() -> None:
    orch, clock, sink, rec_logger, cfg, idle_ms = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    _run_one_row_to_transferred(orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1, idx=0)

    cycle_id_before = orch.snapshot().cycle_id
    assert len(_cycle_start_lines(rec_logger)) == 1

    # No observations for longer than the idle threshold, then one fresh
    # (empty) detection update to trigger the check.
    clock.advance(idle_ms / 1000.0 + 1.0)
    feed(orch, clock, 10_000, [])

    snap = orch.snapshot()
    assert snap.cycle_id == cycle_id_before + 1
    idle_starts = [t for t in _cycle_start_lines(rec_logger) if "idle_timeout" in t]
    assert len(idle_starts) == 1


# ---------------------------------------------------------------------------
# C4 — idle_timeout_blocked_during_transfer
# ---------------------------------------------------------------------------


def test_idle_timeout_blocked_during_transfer() -> None:
    orch, clock, sink, rec_logger, cfg, idle_ms = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id, crossing_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1,
    )
    feed(orch, clock, frame_id, [obs(crossing_front, crossing_front + 50)])
    frame_id += 1
    ack_evt = orch.make_event(EventType.PLC_STOP_ACK, source_frame_id=frame_id)
    orch.submit(ack_evt)
    orch.drain()

    row = next(r for r in orch.snapshot().rows if r.state == RowState.TRANSFER_PENDING)
    cycle_id_before = orch.snapshot().cycle_id
    sink_calls_before = len(sink.calls)

    # Advance well past the idle threshold with no observations, then feed
    # one empty fresh detection update — the in-flight transfer must block
    # the idle timeout entirely.
    clock.advance(idle_ms / 1000.0 + 5.0)
    feed(orch, clock, frame_id + 1000, [])

    snap = orch.snapshot()
    assert snap.cycle_id == cycle_id_before, "an in-flight transfer must never be interrupted by an idle timeout"
    row_after = row_by_track(orch, row.track_id)
    assert row_after.state == RowState.TRANSFER_PENDING
    assert row_after.row_id == row.row_id
    assert len(sink.calls) == sink_calls_before
    assert not rec_logger.has("idle_timeout")


# ---------------------------------------------------------------------------
# C5 — rising_edge_with_no_pending_row
# ---------------------------------------------------------------------------


def test_rising_edge_with_no_pending_row() -> None:
    orch, clock, sink, rec_logger, cfg, _idle_ms = make_orchestrator()
    orch.start_cycle(0)
    transfer_x = 1000.0
    orch.set_transfer_x(transfer_x)

    frame_id, _next_front = create_and_arm_row(
        orch, clock, cfg, transfer_x, start_front=1400.0, frame_id=1,
    )
    before = {(r.row_id, r.track_id, r.state) for r in orch.snapshot().rows}
    cycle_id_before = orch.snapshot().cycle_id

    evt = orch.make_event(EventType.PLC_CLOTH_START, source_frame_id=frame_id)
    orch.submit(evt)
    orch.drain()

    assert rec_logger.has("no_row_awaiting_transfer")
    after = {(r.row_id, r.track_id, r.state) for r in orch.snapshot().rows}
    assert after == before, "no row_id/track_id/state may change on a rejected rising edge"
    assert orch.snapshot().cycle_id == cycle_id_before
    assert len(sink.calls) == 0
