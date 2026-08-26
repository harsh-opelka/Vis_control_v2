"""TransferOrchestrator — the single source of truth for row/transfer state.

Owns row identity, the row lifecycle state machine, and the single call path
to the stop-command sink (wired by MainWindow to the real StopTuchabzug
path). No Qt imports. No OpenCV imports. No I/O of its own — everything it
needs (clock, logger, stop sink) is injected so it is fully unit-testable
with fakes: no Qt, no camera, no PLC, no real clock required.

See viscontrol/core/row_associator.py (association only) and
viscontrol/core/transfer_events.py (the pure data model) for the two
collaborators this class wires together.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from dataclasses import replace

from viscontrol.core.logger import logger as _default_logger
from viscontrol.core.row_associator import RowAssociator
from viscontrol.core.transfer_events import (
    EventType,
    OrchestratorSnapshot,
    RowRecord,
    RowState,
    TransferEvent,
    TransferSnapshot,
)

# Legal (from -> to) transitions, excluding the universal "-> ABANDONED"
# escape hatch (legal from any state — see _transition).
_ALLOWED_TRANSITIONS = {
    (RowState.DETECTED, RowState.ACTIVE),
    (RowState.ACTIVE, RowState.STOP_REQUESTED),
    (RowState.STOP_REQUESTED, RowState.STOPPED),
    (RowState.STOPPED, RowState.TRANSFER_PENDING),
    (RowState.TRANSFER_PENDING, RowState.TRANSFERRING),
    (RowState.TRANSFERRING, RowState.TRANSFERRED),
}

# States in which a row "owns" the transfer — at most one row may ever be in
# one of these simultaneously (see _assert_invariants).
_OWNING_STATES = (
    RowState.STOP_REQUESTED,
    RowState.STOPPED,
    RowState.TRANSFER_PENDING,
    RowState.TRANSFERRING,
)

_TERMINAL_STATES = (RowState.TRANSFERRED, RowState.ABANDONED)

_LEVEL_METHODS = ("debug", "info", "warning", "error", "critical")


class TransferOrchestrator:
    """The single source of truth for transfer/row state.

    Public API (the ONLY way anything interacts with transfer state):
      submit(event), drain(), make_event(...), snapshot(), start_cycle(...),
      shutdown(), set_transfer_x(x).

    ``set_transfer_x`` is a small, necessary addition beyond the literal
    spec: TransferEvent carries no transfer_x field, so the current
    transfer-line x (in the same cloth-local coordinate space as
    ClusterObservation.front_tangent/back_tangent) is tracked as orchestrator
    state, updated by the caller once per frame before drain().
    """

    def __init__(
        self, cfg, stop_command_sink, now_fn=time.monotonic, logger=None,
        cycle_idle_reset_ms: int = 3000,
    ) -> None:
        self._cfg = cfg
        self._stop_command_sink = stop_command_sink
        self._now = now_fn
        self._logger = logger if logger is not None else _default_logger
        self._associator = RowAssociator(cfg)

        self._cycle_id: int = 0
        self._next_event_id: int = 1
        self._next_row_id: int = 1  # NEVER reset within a process run
        self._rows: dict[int, RowRecord] = {}
        self._queue: deque = deque()
        self._lock = threading.RLock()
        self._last_processed_frame_id: int = 0
        self._seen_event_ids: set = set()
        self._display_counter: int = 0  # per cycle, for R1/R2/R3 numbering
        self._transfer_x: float = 0.0
        self._last_raw_cluster_count: int = 0
        # Set at the top of _process_observation and flipped True inside its
        # ambiguous-match loop (EVENT-AMBIGUOUS). Read-only observer state —
        # not used by any association/lifecycle decision; exists solely so
        # callers (e.g. the training-data collector) can tell whether the
        # most recently drained OBSERVATION produced an ambiguous match,
        # without re-deriving it from logs.
        self._last_observation_had_ambiguous: bool = False

        # One physical cloth = one cycle; a cloth stops/restarts once per
        # row (see _process_plc_cloth_start / _process_plc_stop_ack) — those
        # restarts must NEVER call start_cycle(). start_cycle() is reserved
        # for an explicit session start (caller-driven) or this idle-based
        # fallback: no fresh observation for cycle_idle_reset_ms AND no row
        # currently owns the transfer (see _OWNING_STATES) -> assume the
        # previous cloth is done and a new one is about to start.
        self._cycle_idle_reset_s: float = cycle_idle_reset_ms / 1000.0
        self._last_fresh_observation_ts: float = self._now()

    # ---------- public API ----------

    def set_transfer_x(self, transfer_x: float) -> None:
        """Update the current transfer-line x. Call once per frame before
        drain() (see class docstring)."""
        with self._lock:
            self._transfer_x = float(transfer_x)

    def make_event(
        self,
        event_type: EventType,
        source_frame_id: int,
        observations: tuple = (),
        row_id: int | None = None,
    ) -> TransferEvent:
        """Assign the next monotonic event_id and stamp the current cycle_id
        and now_fn(). Thread-safe."""
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            return TransferEvent(
                event_id=event_id,
                cycle_id=self._cycle_id,
                event_type=event_type,
                timestamp=self._now(),
                source_frame_id=source_frame_id,
                observations=tuple(observations),
                row_id=row_id,
            )

    def submit(self, event: TransferEvent) -> None:
        """Thread-safe. Appends to the internal queue. Does NOT process.
        This is what the PLC worker thread (and _run_transfer_orchestrator
        on the GUI thread) calls."""
        with self._lock:
            self._queue.append(event)

    def drain(self) -> None:
        """Processes all queued events in FIFO order under self._lock. MUST
        be called only from the GUI thread inside _on_frame."""
        with self._lock:
            while self._queue:
                event = self._queue.popleft()
                try:
                    self._process_event(event)
                except AssertionError:
                    # _assert_invariants already logged CRITICAL with the
                    # full row table. Do not let it crash the drain loop or
                    # block later (independent) events.
                    pass
                except Exception:  # noqa: BLE001
                    self._logger.exception(
                        "[ORCH] UNCAUGHT-ERROR processing event {}", event
                    )

    @property
    def last_observation_had_ambiguous(self) -> bool:
        """True if the most recently processed OBSERVATION event (inside
        drain()) produced at least one ambiguous match (EVENT-AMBIGUOUS)."""
        with self._lock:
            return self._last_observation_had_ambiguous

    def snapshot(self) -> OrchestratorSnapshot:
        """Returns deep copies. UI reads this and nothing else."""
        with self._lock:
            rows_sorted = sorted(self._rows.values(), key=lambda r: r.display_number)
            rows_copy = tuple(copy.deepcopy(r) for r in rows_sorted)

            def _count(state: RowState) -> int:
                return sum(1 for r in rows_sorted if r.state == state)

            return OrchestratorSnapshot(
                cycle_id=self._cycle_id,
                raw_cluster_count=self._last_raw_cluster_count,
                rows=rows_copy,
                rows_active=_count(RowState.ACTIVE),
                rows_stop_requested=_count(RowState.STOP_REQUESTED),
                rows_transfer_pending=_count(RowState.TRANSFER_PENDING),
                rows_transferring=_count(RowState.TRANSFERRING),
                rows_transferred=_count(RowState.TRANSFERRED),
                rows_abandoned=_count(RowState.ABANDONED),
            )

    def start_cycle(self, source_frame_id: int, reason: str = "session_start") -> None:
        """Increments cycle_id, resets the associator, moves all
        non-terminal rows to ABANDONED (reason="cycle_restart"), logs
        CYCLE-START.

        One physical cloth = one cycle. This must ONLY be called for a
        genuine new-cloth boundary: an explicit caller-driven session start
        (reason="session_start") or the idle-based fallback in
        _process_observation (reason="idle_timeout"). A cloth restarting
        mid-cycle to hand off a row (PLC_CLOTH_START / PLC_STOP_ACK) must
        NEVER reach this method — see _process_plc_cloth_start.
        """
        with self._lock:
            self._cycle_id += 1
            self._display_counter = 0
            self._associator.reset()
            ts = self._now()
            for row in self._existing_rows_sorted():
                if row.state not in _TERMINAL_STATES:
                    old_state = row.state
                    row.state = RowState.ABANDONED
                    row.history.append((ts, old_state, RowState.ABANDONED, "cycle_restart"))
                    self._log(
                        "ROW-TRANSITION", level="info", row=row,
                        from_state=old_state, to_state=RowState.ABANDONED,
                        action="transition", result="ok", reason="cycle_restart",
                    )
            self._last_processed_frame_id = source_frame_id
            self._last_fresh_observation_ts = ts
            self._log(
                "CYCLE-START", level="info", action="start_cycle", result="ok",
                reason=reason, extra_cycle=self._cycle_id, extra_frame=source_frame_id,
            )

    def shutdown(self) -> None:
        """Logs a final summary."""
        with self._lock:
            total = len(self._rows)
            transferred = sum(1 for r in self._rows.values() if r.state == RowState.TRANSFERRED)
            abandoned = sum(1 for r in self._rows.values() if r.state == RowState.ABANDONED)
            self._log(
                "CYCLE-END", level="info", action="shutdown", result="ok",
                reason=f"rows_total={total} transferred={transferred} abandoned={abandoned}",
            )

    # ---------- internals: event dispatch + validation ----------

    def _existing_rows_sorted(self) -> list:
        return [self._rows[rid] for rid in sorted(self._rows.keys())]

    def _process_event(self, event: TransferEvent) -> None:
        # --- validation (applied to EVERY event before processing) ---
        if event.event_id in self._seen_event_ids:
            self._log(
                "EVENT-REJECT", level="warning", event=event,
                reason="duplicate_event_id",
            )
            return
        self._seen_event_ids.add(event.event_id)

        if event.cycle_id != self._cycle_id:
            self._log(
                "OLD-CYCLE-EVENT-BLOCKED", level="warning", event=event,
                reason="stale_cycle",
            )
            return

        if (
            event.event_type == EventType.OBSERVATION
            and event.source_frame_id < self._last_processed_frame_id
        ):
            self._log(
                "OUT-OF-ORDER-EVENT-BLOCKED", level="warning", event=event,
                reason="frame_out_of_order",
            )
            return

        if event.row_id is not None and event.row_id not in self._rows:
            self._log(
                "EVENT-REJECT", level="warning", event=event,
                reason="unknown_row_id",
            )
            return

        # --- dispatch ---
        if event.event_type == EventType.OBSERVATION:
            self._process_observation(event)
        elif event.event_type == EventType.PLC_STOP_ACK:
            self._process_plc_stop_ack(event)
            self._assert_invariants()
        elif event.event_type == EventType.PLC_CLOTH_START:
            self._process_plc_cloth_start(event)
            self._assert_invariants()
        # CYCLE_START / CYCLE_END are not processed here — start_cycle() and
        # shutdown() are direct (unqueued) calls on the orchestrator.

    # ---------- OBSERVATION processing ----------

    def _process_observation(self, event: TransferEvent) -> None:
        self._last_observation_had_ambiguous = False
        # 0. idle-based new-cloth trigger (see start_cycle docstring). Only
        # fires when no row currently owns the transfer — an in-flight
        # transfer must never be interrupted by an idle timeout.
        now = self._now()
        no_owning_row = not any(r.state in _OWNING_STATES for r in self._rows.values())
        if no_owning_row and (now - self._last_fresh_observation_ts) > self._cycle_idle_reset_s:
            self.start_cycle(event.source_frame_id, reason="idle_timeout")
        if event.observations:
            self._last_fresh_observation_ts = now

        self._last_raw_cluster_count = len(event.observations)
        result = self._associator.associate(
            list(event.observations), event.source_frame_id, self._transfer_x,
            self._existing_rows_sorted(),
        )

        # 2. skipped_late — dead at this point, never referenced again.
        for front, back, reason in result.skipped_late:
            self._log(
                "EVENT-REJECT", level="warning", event=event,
                reason=f"skipped_late:{reason}", front=front, back=back,
            )

        # 3. ambiguous matches.
        for obs_index, candidate_row_ids, distances, chosen in result.ambiguous:
            self._last_observation_had_ambiguous = True
            self._log(
                "EVENT-AMBIGUOUS", level="warning", event=event,
                reason=f"obs={obs_index} candidates={candidate_row_ids} "
                       f"distances={distances} chosen={chosen}",
            )

        # 4. apply matches.
        for row_id, (front, back, piece_count) in result.matches.items():
            row = self._rows[row_id]
            row.prev_front_tangent = row.front_tangent
            row.front_tangent = front
            row.back_tangent = back
            row.last_seen_frame = event.source_frame_id
            row.missed_frames = 0
            if front > self._transfer_x:
                row.confirmed_upstream_frames += 1
            self._log(
                "ROW-ASSOCIATE", level="debug", row=row, event=event,
                action="associate", result="ok", front=front, back=back,
            )

        # 5. missed rows.
        for row_id in result.missed_row_ids:
            row = self._rows[row_id]
            row.missed_frames += 1
            self._log(
                "ROW-MISS", level="debug", row=row, event=event,
                action="miss", result="ok", reason=f"missed_frames={row.missed_frames}",
            )

        # 6. DETECTED -> ACTIVE.
        for row in self._existing_rows_sorted():
            if row.state == RowState.DETECTED:
                if (
                    row.confirmed_upstream_frames >= self._cfg.row_arm_min_frames
                    and row.front_tangent > self._transfer_x
                ):
                    self._transition(row, RowState.ACTIVE, "confirmed_upstream", event)

        # 7. tripwire crossing for ACTIVE rows only.
        for row in self._existing_rows_sorted():
            if row.state == RowState.ACTIVE:
                crossed = (
                    row.prev_front_tangent > self._transfer_x
                    and row.front_tangent <= self._transfer_x
                )
                if crossed:
                    self._request_stop(row, event)

        # 8. TRANSFERRING -> TRANSFERRED.
        for row in self._existing_rows_sorted():
            if row.state == RowState.TRANSFERRING:
                timed_out = (
                    row.transfer_started_at is not None
                    and self._now() - row.transfer_started_at
                    > self._cfg.transfer_complete_timeout_s
                )
                exited = row.front_tangent < self._transfer_x - self._cfg.row_exit_margin_px
                if exited or timed_out:
                    reason = "transfer_timeout" if timed_out and not exited else "exit_margin"
                    if self._transition(row, RowState.TRANSFERRED, reason, event):
                        self._log(
                            "TRANSFER-COMPLETE", level="info", row=row, event=event,
                            action="transfer_complete", result="ok", reason=reason,
                        )

        # 9. timeout-driven ABANDONED transitions.
        for row in self._existing_rows_sorted():
            if row.state in (RowState.DETECTED, RowState.ACTIVE):
                # CRITICAL RULE: a detection miss may ONLY abandon a row from
                # DETECTED or ACTIVE — never from an owning state.
                if row.missed_frames > self._cfg.row_max_missed_frames:
                    assert row.state in (RowState.DETECTED, RowState.ACTIVE)
                    was_active = row.state == RowState.ACTIVE
                    if self._transition(row, RowState.ABANDONED, "missed_frames_timeout", event):
                        if was_active:
                            self._logger.warning(
                                "[ORCH] ROW-ABANDONED-BEFORE-CROSSING row={} track={} "
                                "frame={} — a row vanished before crossing the transfer line",
                                row.row_id, row.track_id, event.source_frame_id,
                            )
            elif row.state == RowState.STOP_REQUESTED:
                if (
                    row.stop_requested_at is not None
                    and self._now() - row.stop_requested_at > self._cfg.stop_ack_timeout_s
                ):
                    if self._transition(row, RowState.ABANDONED, "stop_ack_timeout", event):
                        self._logger.error(
                            "[ORCH] STOP-ACK-TIMEOUT row={} track={} stop_evt={} frame={} "
                            "— PLC never acknowledged the stop",
                            row.row_id, row.track_id, row.stop_event_id, event.source_frame_id,
                        )

        # 10. create new rows LAST, so a brand-new row can never be
        # evaluated for a tripwire crossing in the same frame it was
        # created.
        for front, back, piece_count, source_indices in result.new_row_candidates:
            self._create_row(front, back, piece_count, event)

        # 11.
        self._last_processed_frame_id = event.source_frame_id

        # 12.
        self._assert_invariants()

    # ---------- PLC event processing ----------

    def _process_plc_stop_ack(self, event: TransferEvent) -> None:
        candidates = [r for r in self._existing_rows_sorted() if r.state == RowState.STOP_REQUESTED]
        if not candidates:
            self._log(
                "EVENT-REJECT", level="warning", event=event,
                reason="no_row_awaiting_stop_ack",
            )
            return
        if len(candidates) > 1:
            self._log(
                "INVARIANT-VIOLATION", level="critical", event=event,
                reason=f"multiple_rows_awaiting_stop_ack:{[r.row_id for r in candidates]}",
            )
            return
        row = candidates[0]
        self._log(
            "PLC-STOP-ACK", level="info", row=row, event=event,
            action="plc_stop_ack", result="ok",
        )
        self._transition(row, RowState.STOPPED, "plc_stop_ack", event)
        self._transition(row, RowState.TRANSFER_PENDING, "auto_after_stop_ack", event)

    def _process_plc_cloth_start(self, event: TransferEvent) -> None:
        """PLC rising edge (cloth restarts to hand a row off). Does EXACTLY
        ONE thing: the row in TRANSFER_PENDING -> TRANSFERRING. Never calls
        start_cycle(), never abandons/resets any row or tracker — this is a
        mid-cycle event, not a new cycle (one physical cloth = one
        orchestrator cycle; see start_cycle's docstring)."""
        candidates = [r for r in self._existing_rows_sorted() if r.state == RowState.TRANSFER_PENDING]
        if not candidates:
            self._log(
                "EVENT-REJECT", level="warning", event=event,
                reason="no_row_awaiting_transfer",
            )
            return
        if len(candidates) > 1:
            self._log(
                "INVARIANT-VIOLATION", level="critical", event=event,
                reason=f"multiple_rows_awaiting_transfer:{[r.row_id for r in candidates]}",
            )
            return
        row = candidates[0]
        row.transfer_event_id = f"C{self._cycle_id}:R{row.row_id}:XFER"
        if row.snapshot is not None:
            row.snapshot = replace(row.snapshot, transfer_event_id=row.transfer_event_id)
        self._transition(row, RowState.TRANSFERRING, "plc_cloth_start", event)
        self._log(
            "CYCLE-CONTINUE", level="info", row=row, event=event,
            action="cloth_restart", result="ok", reason="row_transfer_started",
        )

    # ---------- row creation ----------

    def _create_row(self, front: float, back: float, piece_count: int, event: TransferEvent) -> RowRecord:
        row_id = self._next_row_id
        self._next_row_id += 1
        self._display_counter += 1
        row = RowRecord(
            row_id=row_id,
            cycle_id=self._cycle_id,
            track_id=row_id,  # association identity == row identity in this model
            display_number=self._display_counter,
            state=RowState.DETECTED,
            front_tangent=front,
            back_tangent=back,
            first_seen_frame=event.source_frame_id,
            last_seen_frame=event.source_frame_id,
            confirmed_upstream_frames=0,
            missed_frames=0,
            prev_front_tangent=front,
            created_at=self._now(),
        )
        self._rows[row_id] = row
        self._log(
            "ROW-CREATE", level="info", row=row, event=event,
            action="create", result="ok", front=front, back=back,
        )
        return row

    # ---------- the only place a stop is ever issued ----------

    def _request_stop(self, row: RowRecord, event: TransferEvent) -> None:
        # 1. Idempotency.
        if row.stop_event_id is not None:
            self._log(
                "STOP-DUPLICATE-BLOCKED", level="warning", row=row, event=event,
                action="request_stop", result="blocked", reason="already_requested",
            )
            return
        if row.state is not RowState.ACTIVE:
            self._log(
                "ILLEGAL-TRANSITION-BLOCKED", level="warning", row=row, event=event,
                from_state=row.state, to_state=RowState.STOP_REQUESTED,
                reason="not_active",
            )
            return
        # 2. Invariant guard — another row already owns the transfer.
        owner = next(
            (r for r in self._existing_rows_sorted() if r.row_id != row.row_id and r.state in _OWNING_STATES),
            None,
        )
        if owner is not None:
            self._log(
                "EVENT-REJECT", level="warning", row=row, event=event,
                reason=f"transfer_already_owned_by_row_{owner.row_id}",
            )
            return
        # 3. Build the immutable snapshot.
        row.stop_event_id = f"C{self._cycle_id}:R{row.row_id}:STOP"
        row.snapshot = TransferSnapshot(
            cycle_id=self._cycle_id, row_id=row.row_id, track_id=row.track_id,
            stop_event_id=row.stop_event_id, transfer_event_id=None,
            created_at=self._now(), created_from_frame_id=event.source_frame_id,
        )
        # 4. Transition, then command — in that order.
        self._transition(row, RowState.STOP_REQUESTED, "tripwire_crossing", event)
        self._log(
            "STOP-COMMAND", level="info", row=row, event=event,
            action="stop_command", result="ok",
        )
        self._stop_command_sink(row.snapshot)

    # ---------- transitions + invariant ----------

    def _transition(self, row: RowRecord, new_state: RowState, reason: str, event: TransferEvent) -> bool:
        old_state = row.state
        legal = new_state == RowState.ABANDONED or (old_state, new_state) in _ALLOWED_TRANSITIONS
        if not legal:
            self._log(
                "ILLEGAL-TRANSITION-BLOCKED", level="warning", row=row, event=event,
                from_state=old_state, to_state=new_state, reason=reason,
            )
            return False
        ts = self._now()
        row.state = new_state
        row.history.append((ts, old_state, new_state, reason))
        if new_state == RowState.STOP_REQUESTED:
            row.stop_requested_at = ts
        elif new_state == RowState.STOPPED:
            row.stopped_at = ts
        elif new_state == RowState.TRANSFERRING:
            row.transfer_started_at = ts
        elif new_state == RowState.TRANSFERRED:
            row.transferred_at = ts
        self._log(
            "ROW-TRANSITION", level="info", row=row, event=event,
            from_state=old_state, to_state=new_state,
            action="transition", result="ok", reason=reason,
        )
        return True

    def _assert_invariants(self) -> None:
        owning = [
            r for r in self._rows.values()
            if r.state in (
                RowState.STOP_REQUESTED, RowState.STOPPED,
                RowState.TRANSFER_PENDING, RowState.TRANSFERRING,
            )
        ]
        if len(owning) > 1:
            table = [(r.row_id, r.state.value) for r in owning]
            self._logger.critical(
                "[ORCH] INVARIANT-VIOLATION ts={:.3f} reason={} full_row_table={}",
                self._now(),
                f"{len(owning)}_rows_own_the_transfer:{table}",
                [(r.row_id, r.state.value) for r in self._existing_rows_sorted()],
            )
            assert len(owning) <= 1, (
                f"INVARIANT VIOLATION: {len(owning)} rows own the transfer: "
                f"{[(r.row_id, r.state) for r in owning]}"
            )

    # ---------- structured logging ----------

    def _log(
        self,
        prefix: str,
        *,
        level: str,
        row: RowRecord | None = None,
        event: TransferEvent | None = None,
        from_state: RowState | str = "-",
        to_state: RowState | str = "-",
        action: str = "-",
        result: str = "-",
        reason: str = "-",
        front: float | None = None,
        back: float | None = None,
        extra_cycle: int | None = None,
        extra_frame: int | None = None,
    ) -> None:
        ts = self._now()
        evt_id = event.event_id if event is not None else "-"
        cycle = (
            extra_cycle if extra_cycle is not None
            else (event.cycle_id if event is not None else self._cycle_id)
        )
        row_id = row.row_id if row is not None else "-"
        disp = f"R{row.display_number}" if row is not None else "-"
        track = row.track_id if row is not None else "-"
        frame = (
            extra_frame if extra_frame is not None
            else (event.source_frame_id if event is not None else "-")
        )
        stop_evt = (row.stop_event_id if row is not None and row.stop_event_id else "-")
        xfer_evt = (row.transfer_event_id if row is not None and row.transfer_event_id else "-")
        front_val = front if front is not None else (row.front_tangent if row is not None else None)
        back_val = back if back is not None else (row.back_tangent if row is not None else None)
        front_s = f"{front_val:.1f}" if front_val is not None else "-"
        back_s = f"{back_val:.1f}" if back_val is not None else "-"
        from_s = from_state.value if isinstance(from_state, RowState) else from_state
        to_s = to_state.value if isinstance(to_state, RowState) else to_state

        line = (
            "[ORCH] {} ts={:.3f} evt={} cycle={} row={} disp={} track={} "
            "from={} to={} action={} result={} reason={} frame={} "
            "stop_evt={} xfer_evt={} front={} back={}"
        ).format(
            prefix, ts, evt_id, cycle, row_id, disp, track,
            from_s, to_s, action, result, reason, frame,
            stop_evt, xfer_evt, front_s, back_s,
        )
        level_name = level.lower()
        if level_name not in _LEVEL_METHODS:
            level_name = "info"
        getattr(self._logger, level_name)(line)
