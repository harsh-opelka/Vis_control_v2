"""Tests for ``viscontrol.core.state_machine``."""

from __future__ import annotations

import pytest

from viscontrol.core.events import State, Verdict
from viscontrol.core.state_machine import StateMachine


def test_initial_state_is_waiting() -> None:
    sm = StateMachine()
    assert sm.state == State.WAITING


def test_pulse_cycle_clean() -> None:
    sm = StateMachine()
    assert sm.handle_tuchabzug_rising()
    assert sm.state == State.TRACKING

    sm.handle_stop_tuchabzug_trigger()
    assert sm.snapshot().stop_tuchabzug is True

    sm.handle_tuchabzug_falling()  # PLC dropped
    sm.handle_inspection_start()
    assert sm.state == State.INSPECTING

    assert sm.handle_pipeline_result(Verdict.OK)  # READY then WAITING
    assert sm.state == State.WAITING
    assert sm.snapshot().pulse_count == 1


def test_pulse_cycle_fault_row_fused() -> None:
    sm = StateMachine()
    sm.handle_tuchabzug_rising()
    sm.handle_tuchabzug_falling()
    sm.handle_inspection_start()
    assert sm.handle_pipeline_result(
        Verdict.FAULT_ROW_FUSED, fault_reason="row_fused"
    )
    assert sm.state == State.FAULT
    snap = sm.snapshot()
    assert snap.fault_active
    assert snap.fault_count == 1
    assert snap.last_fault_reason == "row_fused"


def test_fault_self_clear_after_n_clean_frames() -> None:
    sm = StateMachine(fault_clear_frames=3)
    sm.handle_tuchabzug_rising()
    sm.handle_tuchabzug_falling()
    sm.handle_inspection_start()
    sm.handle_pipeline_result(Verdict.FAULT_UNKNOWN)
    assert sm.state == State.FAULT

    # Two clean frames — still in FAULT.
    sm.handle_clean_frame_in_fault()
    sm.handle_clean_frame_in_fault()
    assert sm.state == State.FAULT
    # Third clean frame -> WAITING.
    assert sm.handle_clean_frame_in_fault()
    assert sm.state == State.WAITING
    assert sm.snapshot().fault_active is False


def test_dirty_frame_in_fault_resets_counter() -> None:
    sm = StateMachine(fault_clear_frames=3)
    sm.handle_tuchabzug_rising()
    sm.handle_tuchabzug_falling()
    sm.handle_inspection_start()
    sm.handle_pipeline_result(Verdict.FAULT_UNKNOWN)
    sm.handle_clean_frame_in_fault()
    sm.handle_clean_frame_in_fault()
    sm.handle_dirty_frame_in_fault()
    sm.handle_clean_frame_in_fault()  # only 1 consecutive again
    assert sm.state == State.FAULT


def test_info_verdicts_do_not_fault() -> None:
    sm = StateMachine()
    sm.handle_tuchabzug_rising()
    sm.handle_tuchabzug_falling()
    sm.handle_inspection_start()
    sm.handle_pipeline_result(Verdict.INFO_COLUMN_FUSED)
    assert sm.state == State.WAITING
    assert sm.snapshot().fault_count == 0


def test_rising_edge_during_tracking_ignored() -> None:
    sm = StateMachine()
    sm.handle_tuchabzug_rising()
    assert sm.handle_tuchabzug_rising() is False  # already TRACKING
    assert sm.state == State.TRACKING


def test_inspection_start_only_from_tracking() -> None:
    sm = StateMachine()
    assert sm.handle_inspection_start() is False
    assert sm.state == State.WAITING


def test_pipeline_result_outside_inspecting_ignored() -> None:
    sm = StateMachine()
    sm.handle_pipeline_result(Verdict.FAULT_UNKNOWN)
    assert sm.state == State.WAITING


def test_emergency_stop_forces_fault() -> None:
    sm = StateMachine()
    sm.handle_tuchabzug_rising()
    assert sm.handle_emergency_stop()
    assert sm.state == State.FAULT
    assert sm.snapshot().last_fault_reason == "emergency_stop"


def test_service_entry_and_exit() -> None:
    sm = StateMachine()
    sm.handle_tuchabzug_rising()
    sm.enter_service()
    assert sm.state == State.SERVICE
    # Emergency stop from SERVICE is ignored.
    assert sm.handle_emergency_stop() is False
    assert sm.state == State.SERVICE
    sm.exit_service()
    assert sm.state == State.WAITING


def test_on_change_callback_fires() -> None:
    sm = StateMachine()
    events: list[tuple[State, State]] = []
    sm.on_change(lambda old, new: events.append((old, new)))
    sm.handle_tuchabzug_rising()
    sm.handle_tuchabzug_falling()
    sm.handle_inspection_start()
    sm.handle_pipeline_result(Verdict.OK)
    assert events[0] == (State.WAITING, State.TRACKING)
    assert events[1] == (State.TRACKING, State.INSPECTING)
    # Two transitions from INSPECTING: -> READY -> WAITING.
    assert events[-2] == (State.INSPECTING, State.READY)
    assert events[-1] == (State.READY, State.WAITING)


def test_listener_exception_does_not_break_sm() -> None:
    sm = StateMachine()
    sm.on_change(lambda old, new: (_ for _ in ()).throw(RuntimeError("boom")))
    # Must not raise.
    assert sm.handle_tuchabzug_rising()
    assert sm.state == State.TRACKING


def test_force_reset_helpers() -> None:
    sm = StateMachine()
    sm.handle_tuchabzug_rising()
    sm.handle_tuchabzug_falling()
    sm.handle_inspection_start()
    sm.handle_pipeline_result(Verdict.FAULT_UNKNOWN)
    sm.force_reset_to_waiting()
    assert sm.state == State.WAITING
    assert sm.snapshot().fault_active is False

    sm.force_reset_to_tracking()
    assert sm.state == State.TRACKING


def test_fault_clear_frames_validation() -> None:
    with pytest.raises(ValueError):
        StateMachine(fault_clear_frames=0)
