"""Central state machine for VisControl.

The state machine owns the transitions described in the build spec and never
touches Qt directly. It exposes:

- ``handle_*`` methods that callers invoke on real-world events
- a list of ``on_change`` callbacks that fire whenever the state actually
  changes (so the UI can pick up the new state via Qt queued signals)

The fault self-clear logic counts consecutive clean frames in INSPECTING/READY;
once it reaches ``fault_clear_frames`` we drop back to WAITING. We don't try to
self-clear from FAULT directly — the spec says FAULT clears only after N
consecutive clean frames, which only the inspection path can produce.

FIX 4: in production, MainWindow does NOT use the clean-frame self-clear path
at all — :meth:`acknowledge_fault` is the only way FAULT clears there, paired
1:1 with the PLC's ext_error_quit so the UI and the PLC's latched ext_error
are always in sync. The clean-frame self-clear (:meth:`handle_clean_frame_in_fault`)
remains the demo-mode clear path, since demo has no PLC ack signal.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from viscontrol.core.events import State, Verdict


@dataclass
class StateContext:
    """Mutable snapshot of state-machine-relevant inputs.

    Kept separate from :class:`StateMachine` so callers can read the latest
    snapshot without holding the SM lock.
    """

    state: State = State.WAITING
    previous_state: State = State.WAITING
    tuchabzug_running: bool = False
    stop_tuchabzug: bool = False
    fault_active: bool = False
    consecutive_clean_frames: int = 0
    last_fault_reason: str = ""
    pulse_count: int = 0
    fault_count: int = 0
    extras: dict = field(default_factory=dict)


OnChangeCallback = Callable[[State, State], None]


class StateMachine:
    """Threadsafe state machine.

    Transitions are guarded by a single ``RLock``; all reads should go through
    :meth:`snapshot` so the UI gets a consistent view.
    """

    def __init__(self, *, fault_clear_frames: int = 5) -> None:
        if fault_clear_frames < 1:
            raise ValueError("fault_clear_frames must be >= 1")
        self._fault_clear_frames = fault_clear_frames
        self._lock = threading.RLock()
        self._ctx = StateContext()
        self._listeners: list[OnChangeCallback] = []

    # ---------- listeners / snapshot ----------

    def on_change(self, cb: OnChangeCallback) -> None:
        """Register a callback fired on every (old -> new) state change."""
        with self._lock:
            self._listeners.append(cb)

    def snapshot(self) -> StateContext:
        """Return a copy of the current context (safe to read without lock)."""
        with self._lock:
            return StateContext(
                state=self._ctx.state,
                previous_state=self._ctx.previous_state,
                tuchabzug_running=self._ctx.tuchabzug_running,
                stop_tuchabzug=self._ctx.stop_tuchabzug,
                fault_active=self._ctx.fault_active,
                consecutive_clean_frames=self._ctx.consecutive_clean_frames,
                last_fault_reason=self._ctx.last_fault_reason,
                pulse_count=self._ctx.pulse_count,
                fault_count=self._ctx.fault_count,
                extras=dict(self._ctx.extras),
            )

    @property
    def state(self) -> State:
        with self._lock:
            return self._ctx.state

    # ---------- transitions ----------

    def _transition(self, new_state: State) -> bool:
        """Set state and notify listeners if it actually changed. Returns True if changed."""
        old = self._ctx.state
        if old == new_state:
            return False
        self._ctx.previous_state = old
        self._ctx.state = new_state
        listeners = list(self._listeners)
        # Notify outside the lock so a listener can't deadlock by re-entering.
        for cb in listeners:
            try:
                cb(old, new_state)
            except Exception:
                # Listeners must not break the SM. Swallow and rely on app logger.
                pass
        return True

    def handle_tuchabzug_rising(self) -> bool:
        """Either the PLC raised ``TuchabzugRunning`` (Prod) or the user pressed
        Simulate Pulse (Demo). Either way, kick off a tracking cycle.

        Returns True if a state change occurred.
        """
        with self._lock:
            self._ctx.tuchabzug_running = True
            if self._ctx.state in (State.WAITING, State.READY):
                return self._transition(State.TRACKING)
            # In TRACKING/INSPECTING/FAULT/SERVICE we ignore the rising edge.
            return False

    def handle_tuchabzug_falling(self) -> bool:
        """``TuchabzugRunning`` went false. Caller is expected to apply the
        ``delay_after_pull_ms`` settling delay before calling
        :meth:`handle_inspection_start`.
        """
        with self._lock:
            self._ctx.tuchabzug_running = False
            self._ctx.stop_tuchabzug = False
            # We don't auto-transition here; the settling delay belongs to the caller.
            return False

    def handle_inspection_start(self) -> bool:
        """Caller has waited the settling delay and wants to switch to INSPECTING."""
        with self._lock:
            if self._ctx.state == State.TRACKING:
                return self._transition(State.INSPECTING)
            return False

    def handle_stop_tuchabzug_trigger(self) -> bool:
        """A row crossed the transfer line; assert StopTuchabzug.

        Fires whenever TuchabzugRunning is True, regardless of internal state,
        so the tripwire works during FAULT as well as TRACKING.
        """
        with self._lock:
            if self._ctx.tuchabzug_running:
                self._ctx.stop_tuchabzug = True
            return False

    def handle_stop_tuchabzug_clear(self) -> None:
        """Tripwire falling edge: the line is clear, re-arm for the next row."""
        with self._lock:
            if self._ctx.tuchabzug_running:
                self._ctx.stop_tuchabzug = False

    def raise_belt_fault(self, fault_reason: str = "") -> bool:
        """Raise a belt-side fault from the concurrent detection path.

        Works from TRACKING or WAITING — not gated on INSPECTING state.
        Returns True if a FAULT transition occurred; False if already in FAULT
        or SERVICE (idempotent for the same fault row).
        """
        with self._lock:
            if self._ctx.state in (State.FAULT, State.SERVICE):
                return False
            self._ctx.fault_active = True
            self._ctx.fault_count += 1
            self._ctx.last_fault_reason = fault_reason
            self._ctx.consecutive_clean_frames = 0
            return self._transition(State.FAULT)

    def handle_pipeline_result(self, verdict: Verdict, *, fault_reason: str = "") -> bool:
        """Feed an INSPECTING-time verdict back into the SM.

        - Fault verdicts -> FAULT (and bump ``fault_count``).
        - OK -> READY -> WAITING immediately (READY is just a UI breadcrumb).
        - INFO_* verdicts behave like OK from the SM's perspective.
        """
        with self._lock:
            if self._ctx.state != State.INSPECTING:
                # Late result — likely a frame from before a reset. Ignore.
                return False
            if verdict.is_fault:
                self._ctx.fault_active = True
                self._ctx.fault_count += 1
                self._ctx.last_fault_reason = fault_reason or verdict.value
                self._ctx.consecutive_clean_frames = 0
                return self._transition(State.FAULT)
            # Clean / informational.
            self._ctx.pulse_count += 1
            self._ctx.consecutive_clean_frames = 0  # reset for next FAULT cycle
            self._transition(State.READY)
            return self._transition(State.WAITING)

    def handle_clean_frame_in_fault(self) -> bool:
        """A post-FAULT classifier pass returned a clean image.

        After ``fault_clear_frames`` consecutive clean frames we drop back to
        WAITING. We do not count clean frames received outside FAULT — the
        counter resets in :meth:`handle_pipeline_result`.
        """
        with self._lock:
            if self._ctx.state != State.FAULT:
                return False
            self._ctx.consecutive_clean_frames += 1
            if self._ctx.consecutive_clean_frames >= self._fault_clear_frames:
                self._ctx.fault_active = False
                self._ctx.consecutive_clean_frames = 0
                return self._transition(State.WAITING)
            return False

    def handle_dirty_frame_in_fault(self) -> None:
        """A post-FAULT classifier pass still sees defects. Reset the counter."""
        with self._lock:
            if self._ctx.state == State.FAULT:
                self._ctx.consecutive_clean_frames = 0

    def acknowledge_fault(self) -> bool:
        """Operator acknowledge (FIX 4): unconditionally clears the latched
        fault, regardless of belt cleanliness.

        This is the production fault-clear path — paired 1:1 with the PLC's
        ext_error_quit rising edge (see MainWindow._on_plc_error_quit), so the
        UI FaultActive indicator and the PLC's ext_error latch clear together,
        immediately. Distinct from :meth:`handle_clean_frame_in_fault`, which
        is the demo/self-clear path (no PLC ack exists in demo).

        Returns True if a FAULT -> WAITING transition occurred.
        """
        with self._lock:
            if self._ctx.state != State.FAULT:
                return False
            self._ctx.fault_active = False
            self._ctx.consecutive_clean_frames = 0
            return self._transition(State.WAITING)

    def handle_emergency_stop(self) -> bool:
        """Production-mode panic: forced FAULT from any non-SERVICE state."""
        with self._lock:
            if self._ctx.state == State.SERVICE:
                return False
            self._ctx.fault_active = True
            self._ctx.last_fault_reason = "emergency_stop"
            self._ctx.consecutive_clean_frames = 0
            return self._transition(State.FAULT)

    def enter_service(self) -> bool:
        """User authenticated into SERVICE. Stops detection-driven transitions."""
        with self._lock:
            self._ctx.stop_tuchabzug = False
            return self._transition(State.SERVICE)

    def exit_service(self) -> bool:
        """Leave SERVICE — always lands in WAITING per spec."""
        with self._lock:
            if self._ctx.state != State.SERVICE:
                return False
            self._ctx.fault_active = False
            self._ctx.consecutive_clean_frames = 0
            return self._transition(State.WAITING)

    def force_reset_to_waiting(self) -> bool:
        """Startup helper — used by main.py when ``TuchabzugRunning`` reads false."""
        with self._lock:
            self._ctx.fault_active = False
            self._ctx.stop_tuchabzug = False
            self._ctx.consecutive_clean_frames = 0
            return self._transition(State.WAITING)

    def force_reset_to_tracking(self) -> bool:
        """Startup helper — used when we boot mid-pulse (TuchabzugRunning=true)."""
        with self._lock:
            self._ctx.tuchabzug_running = True
            self._ctx.fault_active = False
            self._ctx.consecutive_clean_frames = 0
            return self._transition(State.TRACKING)
