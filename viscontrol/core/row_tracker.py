"""Groups stable PieceTracks into physical rows using a Y-exclusivity
constraint, with hysteresis on every membership change.

Consumes PieceTrack objects only (see viscontrol/core/piece_tracker.py) —
never RawDetection, never a proximity cluster. Two dough pieces cannot
physically occupy the same Y position at the same time; if two pieces are
closer than ~0.8 diameters in center_y they cannot be in the same row
regardless of how close their tangent_x values are. This RELATIVE
constraint (never a calibrated/learned Y lane) is what lets
``row_x_tolerance_px`` stay deliberately loose — see RowTracker._propose_groups.

No Qt. No cv2. No PySide6.
"""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass
from enum import Enum


class RowGroupState(str, Enum):
    CANDIDATE = "CANDIDATE"
    STABILIZING = "STABILIZING"
    CONFIRMED = "CONFIRMED"
    LOCKED_FOR_TRANSFER = "LOCKED_FOR_TRANSFER"
    RETIRED = "RETIRED"


@dataclass
class RowGroup:
    row_id: int  # PERMANENT. Never reused.
    cycle_id: int
    state: RowGroupState
    member_piece_ids: set  # set[int]
    membership_version: int  # increments on every accepted change
    front_tangent: float  # min tangent_x over members (smoothed)
    back_tangent: float  # max tangent_x over members (smoothed)
    prev_front_tangent: float
    center_y_values: list  # for logging/diagnostics
    created_frame: int
    confirmed_frame: int | None
    locked_frame: int | None
    stable_updates: int  # consecutive fresh updates with stable membership
    display_number: int
    locked_snapshot: object | None = None  # LockedRowSnapshot


@dataclass(frozen=True)
class LockedRowSnapshot:
    """Immutable. Once created, membership can never change."""

    cycle_id: int
    row_id: int
    member_piece_ids: tuple
    front_tangent: float
    back_tangent: float
    piece_count: int
    membership_version: int
    confirmed_frame: int
    locked_frame: int


@dataclass(frozen=True)
class StableRowObservation:
    """What the TransferOrchestrator receives. Replaces ClusterObservation."""

    row_id: int  # STABLE — orchestrator matches on this EXACTLY
    front_tangent: float
    back_tangent: float
    piece_count: int
    state: RowGroupState
    membership_version: int
    locked_snapshot: object | None


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


class RowTracker:
    """Groups PieceTracks into physical rows and holds membership with
    hysteresis (see module docstring and the CASE A-F reconciliation logic
    in :meth:`_reconcile`)."""

    def __init__(self, cfg, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger
        self._rows: dict[int, RowGroup] = {}
        self._next_row_id: int = 1  # PERMANENT — never reset within a process run
        self._cycle_id: int = 0
        self._display_counter: int = 0

        self._pending_add: dict[int, dict] = {}
        self._pending_remove: dict[int, dict] = {}
        self._pending_merge: dict[tuple, dict] = {}
        self._pending_split: dict[int, dict] = {}

        self._track_lookup: dict[int, object] = {}
        self._confirmed_owner: dict[int, int] = {}

    def reset(self, cycle_id: int) -> None:
        self._rows = {}
        self._cycle_id = cycle_id
        self._display_counter = 0
        self._pending_add = {}
        self._pending_remove = {}
        self._pending_merge = {}
        self._pending_split = {}

    # ---------- public API ----------

    def update(
        self, tracks: list, frame_id: int, transfer_x: float, raw_det_count: int | None = None,
    ) -> list:
        self._track_lookup = {t.piece_id: t for t in tracks}
        self._confirmed_owner = {}
        for row in self._rows.values():
            if row.state in (RowGroupState.CONFIRMED, RowGroupState.LOCKED_FOR_TRANSFER):
                for pid in row.member_piece_ids:
                    self._confirmed_owner[pid] = row.row_id

        proposed_groups = self._propose_groups(tracks, frame_id)
        self._reconcile(proposed_groups, frame_id, transfer_x)
        self._refresh_geometry()
        self._advance_lifecycle(frame_id, transfer_x)

        observations = []
        for row in sorted(self._rows.values(), key=lambda r: r.row_id):
            if row.state in (RowGroupState.CONFIRMED, RowGroupState.LOCKED_FOR_TRANSFER):
                observations.append(StableRowObservation(
                    row_id=row.row_id, front_tangent=row.front_tangent,
                    back_tangent=row.back_tangent, piece_count=len(row.member_piece_ids),
                    state=row.state, membership_version=row.membership_version,
                    locked_snapshot=row.locked_snapshot,
                ))

        self._log_summary(frame_id, raw_det_count, len(tracks))
        return observations

    def snapshot(self) -> list:
        """Read-only deep copies of every non-retired RowGroup, for display
        only (see camera_view.py's per-RowGroupState overlay) — includes
        CANDIDATE/STABILIZING rows that update() never emits as
        StableRowObservation. Never mutated by the caller."""
        return [
            copy.deepcopy(r) for r in sorted(self._rows.values(), key=lambda r: r.row_id)
            if r.state != RowGroupState.RETIRED
        ]

    # ---------- 2a. grouping proposal (advisory) ----------

    def _propose_groups(self, tracks: list, frame_id: int) -> list[list]:
        live = [t for t in tracks if t.fresh_observations >= 1]
        if not live:
            return []

        diam_median = statistics.median(2.0 * t.radius for t in live)
        if self._cfg.row_y_exclusion_px:
            y_excl = float(self._cfg.row_y_exclusion_px)
        else:
            y_excl = self._cfg.row_y_exclusion_factor * diam_median

        ordered = sorted(live, key=lambda t: t.tangent_x, reverse=True)
        groups: list[list] = []
        current: list = [ordered[0]]

        for track in ordered[1:]:
            group_min = min(t.tangent_x for t in current)
            group_max = max(t.tangent_x for t in current)
            if track.tangent_x < group_min:
                x_gap = group_min - track.tangent_x
            elif track.tangent_x > group_max:
                x_gap = track.tangent_x - group_max
            else:
                x_gap = 0.0
            min_dy = min(abs(track.center_y - t.center_y) for t in current)
            y_conflict = min_dy < y_excl
            span_after = max(group_max, track.tangent_x) - min(group_min, track.tangent_x)

            if x_gap <= self._cfg.row_x_tolerance_px and not y_conflict and span_after <= self._cfg.row_max_span_px:
                current.append(track)
                continue

            if y_conflict:
                reason = "y_conflict"
            elif x_gap > self._cfg.row_x_tolerance_px:
                reason = "x_gap"
            else:
                reason = "span"
            self._log(
                "ROW-GROUP-REJECT", level="debug", frame=frame_id, piece=track.piece_id,
                group_rows=[t.piece_id for t in current], reason=reason,
                x_gap=x_gap, min_dy=min_dy, y_excl=y_excl,
            )
            groups.append(current)
            current = [track]
        groups.append(current)
        return groups

    # ---------- 2b. reconciliation (hysteresis) ----------

    def _reconcile(self, proposed_groups: list[list], frame_id: int, transfer_x: float) -> None:
        proposals = [frozenset(t.piece_id for t in g) for g in proposed_groups]
        live_rows = [r for r in self._rows.values() if r.state != RowGroupState.RETIRED]

        row_best: dict[int, tuple[int, float]] = {}
        for row in live_rows:
            members = frozenset(row.member_piece_ids)
            best_idx, best_score = None, 0.0
            for pi, pset in enumerate(proposals):
                score = _jaccard(members, pset)
                if score > 0 and (best_idx is None or score > best_score):
                    best_idx, best_score = pi, score
            if best_idx is not None:
                row_best[row.row_id] = (best_idx, best_score)

        proposal_best_row: dict[int, tuple[int, float]] = {}
        for pi, pset in enumerate(proposals):
            best_row_id, best_score = None, 0.0
            for row in live_rows:
                score = _jaccard(frozenset(row.member_piece_ids), pset)
                if score > 0 and (best_row_id is None or score > best_score):
                    best_row_id, best_score = row.row_id, score
            if best_row_id is not None:
                proposal_best_row[pi] = (best_row_id, best_score)

        proposal_claims: dict[int, list[int]] = {}
        for row_id, (pi, _score) in row_best.items():
            proposal_claims.setdefault(pi, []).append(row_id)

        row_claims: dict[int, list[int]] = {}
        for pi, (row_id, _score) in proposal_best_row.items():
            row_claims.setdefault(row_id, []).append(pi)

        handled_rows: set[int] = set()
        handled_proposals: set[int] = set()
        touched_merge_keys: set[tuple] = set()
        touched_split_rows: set[int] = set()

        # --- CASE D: MERGE — two+ rows both best-match the same proposal ---
        for pi, row_ids in proposal_claims.items():
            if len(row_ids) >= 2:
                row_ids_sorted = sorted(row_ids)
                touched_merge_keys.add(tuple(row_ids_sorted))
                self._handle_merge_candidate(row_ids_sorted, proposals[pi], frame_id, transfer_x)
                handled_rows.update(row_ids_sorted)
                handled_proposals.add(pi)

        # --- CASE E: SPLIT — one row is best-matched by two+ proposals ---
        for row_id, pis in row_claims.items():
            if row_id in handled_rows:
                continue
            if len(pis) >= 2:
                pis_sorted = sorted(pis)
                touched_split_rows.add(row_id)
                self._handle_split_candidate(row_id, pis_sorted, proposals, frame_id, transfer_x)
                handled_rows.add(row_id)
                handled_proposals.update(pis_sorted)

        # --- CASE A/B/C: clean 1:1 pairings ---
        for row in live_rows:
            if row.row_id in handled_rows:
                continue
            if row.row_id not in row_best:
                continue  # no matching proposal this frame
            pi, _score = row_best[row.row_id]
            if pi in handled_proposals:
                continue
            handled_proposals.add(pi)
            handled_rows.add(row.row_id)
            self._handle_one_to_one(row, proposals[pi], frame_id)

        # --- CASE F: leftover proposals with no matching row ---
        for pi, pset in enumerate(proposals):
            if pi in handled_proposals:
                continue
            self._create_candidate_row(pset, proposed_groups[pi], frame_id)

        # Evidence for merge/split is required to be CONSECUTIVE (see CASE
        # D/E spec): any pending entry not re-proposed THIS frame has had
        # its streak broken and must reset, not silently keep accumulating
        # whenever the same row pair/row happens to recur later.
        for key in list(self._pending_merge.keys()):
            if key not in touched_merge_keys:
                self._log_evidence_reset(frame_id, key, "merge", self._pending_merge[key]["count"])
                del self._pending_merge[key]
        for row_id in list(self._pending_split.keys()):
            if row_id not in touched_split_rows:
                self._log_evidence_reset(frame_id, row_id, "split", self._pending_split[row_id]["count"])
                del self._pending_split[row_id]

    def _handle_one_to_one(self, row: RowGroup, pset: frozenset, frame_id: int) -> None:
        cfg = self._cfg
        members = frozenset(row.member_piece_ids)

        if row.state == RowGroupState.LOCKED_FOR_TRANSFER:
            # Membership is FROZEN forever — no additions, no removals, ever.
            result = "unchanged" if pset == members else "locked_immutable"
            if pset == members:
                row.stable_updates += 1
            self._log(
                "ROW-MEMBERSHIP", level="debug", frame=frame_id, row=row.row_id,
                state=row.state.value, members=sorted(row.member_piece_ids),
                version=row.membership_version, result=result, reason="-",
            )
            return

        if pset == members:
            row.stable_updates += 1
            self._pending_add.pop(row.row_id, None)
            self._pending_remove.pop(row.row_id, None)
            self._log(
                "ROW-MEMBERSHIP", level="debug", frame=frame_id, row=row.row_id,
                state=row.state.value, members=sorted(row.member_piece_ids),
                version=row.membership_version, result="unchanged", reason="-",
            )
            return

        added = pset - members
        removed = members - pset
        row.stable_updates = 0

        if added:
            self._handle_add_evidence(row, added, frame_id)
        else:
            self._pending_add.pop(row.row_id, None)

        if removed:
            self._handle_remove_evidence(row, removed, frame_id)
        else:
            self._pending_remove.pop(row.row_id, None)

    def _handle_add_evidence(self, row: RowGroup, added: frozenset, frame_id: int) -> None:
        cfg = self._cfg
        pending = self._pending_add.get(row.row_id)
        if pending and pending["piece_ids"] == added:
            pending["count"] += 1
        else:
            if pending:
                self._log_evidence_reset(frame_id, row.row_id, "add", pending["count"])
            pending = {"piece_ids": added, "count": 1}
            self._pending_add[row.row_id] = pending

        blocked = any(
            self._confirmed_owner.get(pid) not in (None, row.row_id) for pid in added
        )
        if blocked:
            self._log(
                "ROW-MEMBERSHIP", level="debug", frame=frame_id, row=row.row_id,
                state=row.state.value, members=sorted(row.member_piece_ids),
                version=row.membership_version, result="add_blocked",
                reason="piece_owned_by_other_confirmed_row",
            )
            return

        if pending["count"] >= cfg.row_add_evidence_frames:
            row.member_piece_ids = set(row.member_piece_ids) | set(added)
            row.membership_version += 1
            del self._pending_add[row.row_id]
            self._log(
                "ROW-ASSIGNMENT-CHANGED", level="info", frame=frame_id,
                piece=sorted(added), prev_row="-", proposed_row=row.row_id,
                result_row=row.row_id, evidence=(cfg.row_add_evidence_frames, cfg.row_add_evidence_frames),
                score=1.0, reason="add_accepted",
            )
        else:
            self._log(
                "ROW-MEMBERSHIP", level="debug", frame=frame_id, row=row.row_id,
                state=row.state.value, members=sorted(row.member_piece_ids),
                version=row.membership_version, result="add_pending",
                reason=f"{pending['count']}/{cfg.row_add_evidence_frames}",
            )

    def _handle_remove_evidence(self, row: RowGroup, removed: frozenset, frame_id: int) -> None:
        cfg = self._cfg
        pending = self._pending_remove.get(row.row_id)
        if pending and pending["piece_ids"] == removed:
            pending["count"] += 1
        else:
            if pending:
                self._log_evidence_reset(frame_id, row.row_id, "remove", pending["count"])
            pending = {"piece_ids": removed, "count": 1}
            self._pending_remove[row.row_id] = pending

        if pending["count"] >= cfg.row_remove_evidence_frames:
            row.member_piece_ids = set(row.member_piece_ids) - set(removed)
            row.membership_version += 1
            del self._pending_remove[row.row_id]
            self._log(
                "ROW-ASSIGNMENT-CHANGED", level="info", frame=frame_id,
                piece=sorted(removed), prev_row=row.row_id, proposed_row="-",
                result_row="-", evidence=(cfg.row_remove_evidence_frames, cfg.row_remove_evidence_frames),
                score=1.0, reason="remove_accepted",
            )
        else:
            self._log(
                "ROW-MEMBERSHIP", level="debug", frame=frame_id, row=row.row_id,
                state=row.state.value, members=sorted(row.member_piece_ids),
                version=row.membership_version, result="remove_pending",
                reason=f"{pending['count']}/{cfg.row_remove_evidence_frames}",
            )

    def _is_locked_or_near_lock(self, row: RowGroup, transfer_x: float) -> bool:
        cfg = self._cfg
        if row.state == RowGroupState.LOCKED_FOR_TRANSFER:
            return True
        return row.front_tangent <= transfer_x + cfg.row_lock_margin_px

    def _handle_merge_candidate(
        self, row_ids: list[int], pset: frozenset, frame_id: int, transfer_x: float,
    ) -> None:
        cfg = self._cfg
        key = tuple(row_ids)
        rows = [self._rows[rid] for rid in row_ids]

        if any(self._is_locked_or_near_lock(r, transfer_x) for r in rows):
            self._pending_merge.pop(key, None)
            self._log(
                "ROW-MERGE-REJECTED", level="warning", frame=frame_id, rows=list(row_ids),
                reason="locked",
            )
            return

        union_members = frozenset().union(*(frozenset(r.member_piece_ids) for r in rows))
        score_gain = _jaccard(union_members, pset)

        pending = self._pending_merge.get(key)
        if pending:
            pending["count"] += 1
            pending["score"] = score_gain
        else:
            pending = {"count": 1, "score": score_gain}
            self._pending_merge[key] = pending

        self._log(
            "ROW-MERGE-PROPOSED", level="debug", frame=frame_id, rows=list(row_ids),
            evidence=(pending["count"], cfg.row_merge_evidence_frames), score_gain=score_gain,
        )

        if pending["count"] >= cfg.row_merge_evidence_frames and score_gain > cfg.row_merge_min_score_gain:
            survivor_id = min(row_ids)
            retired_ids = [rid for rid in row_ids if rid != survivor_id]
            survivor = self._rows[survivor_id]
            survivor.member_piece_ids = set(pset)
            survivor.membership_version += 1
            for rid in retired_ids:
                self._rows[rid].state = RowGroupState.RETIRED
            del self._pending_merge[key]
            self._log(
                "ROW-MERGE-ACCEPTED", level="info", frame=frame_id, survivor=survivor_id,
                retired=retired_ids, members=sorted(survivor.member_piece_ids),
                version=survivor.membership_version,
            )

    def _handle_split_candidate(
        self, row_id: int, proposal_indices: list[int], proposals: list[frozenset],
        frame_id: int, transfer_x: float,
    ) -> None:
        cfg = self._cfg
        row = self._rows[row_id]
        psets = [proposals[pi] for pi in proposal_indices]

        if self._is_locked_or_near_lock(row, transfer_x):
            self._pending_split.pop(row_id, None)
            self._log("ROW-SPLIT-REJECTED", level="warning", frame=frame_id, row=row_id, reason="locked")
            return

        groups_sig = tuple(sorted(tuple(sorted(p)) for p in psets))
        pending = self._pending_split.get(row_id)
        if pending and pending["groups"] == groups_sig:
            pending["count"] += 1
        else:
            if pending:
                self._log_evidence_reset(frame_id, row_id, "split", pending["count"])
            pending = {"groups": groups_sig, "count": 1}
            self._pending_split[row_id] = pending

        self._log(
            "ROW-SPLIT-PROPOSED", level="debug", frame=frame_id, row=row_id,
            groups=groups_sig, evidence=(pending["count"], cfg.row_split_evidence_frames),
        )

        if pending["count"] >= cfg.row_split_evidence_frames:
            fronts = [self._group_front_tangent(p) for p in psets]
            downstream_idx = min(range(len(psets)), key=lambda i: fronts[i])
            kept_members = psets[downstream_idx]
            new_members = frozenset().union(
                *(psets[i] for i in range(len(psets)) if i != downstream_idx)
            )
            row.member_piece_ids = set(kept_members)
            row.membership_version += 1
            new_row = self._create_row_from_split(new_members, frame_id)
            del self._pending_split[row_id]
            self._log(
                "ROW-SPLIT-ACCEPTED", level="info", frame=frame_id, row=row_id,
                kept_members=sorted(kept_members), new_row=new_row.row_id,
                new_members=sorted(new_members),
            )

    def _create_candidate_row(self, pset: frozenset, group_tracks: list, frame_id: int) -> RowGroup:
        row_id = self._next_row_id
        self._next_row_id += 1
        self._display_counter += 1
        front = min(t.tangent_x for t in group_tracks)
        back = max(t.tangent_x for t in group_tracks)
        ys = [t.center_y for t in group_tracks]
        row = RowGroup(
            row_id=row_id, cycle_id=self._cycle_id, state=RowGroupState.CANDIDATE,
            member_piece_ids=set(pset), membership_version=1,
            front_tangent=front, back_tangent=back, prev_front_tangent=front,
            center_y_values=ys, created_frame=frame_id, confirmed_frame=None,
            locked_frame=None, stable_updates=0, display_number=self._display_counter,
        )
        self._rows[row_id] = row
        self._log(
            "ROW-CANDIDATE-CREATE", level="info", frame=frame_id, row=row_id,
            disp=row.display_number, members=sorted(pset), front=front, back=back,
            ys=[round(y, 1) for y in ys],
        )
        return row

    def _create_row_from_split(self, member_piece_ids: frozenset, frame_id: int) -> RowGroup:
        row_id = self._next_row_id
        self._next_row_id += 1
        self._display_counter += 1
        member_tracks = [self._track_lookup[pid] for pid in member_piece_ids if pid in self._track_lookup]
        front = min((t.tangent_x for t in member_tracks), default=0.0)
        back = max((t.tangent_x for t in member_tracks), default=0.0)
        ys = [t.center_y for t in member_tracks]
        row = RowGroup(
            row_id=row_id, cycle_id=self._cycle_id, state=RowGroupState.CANDIDATE,
            member_piece_ids=set(member_piece_ids), membership_version=1,
            front_tangent=front, back_tangent=back, prev_front_tangent=front,
            center_y_values=ys, created_frame=frame_id, confirmed_frame=None,
            locked_frame=None, stable_updates=0, display_number=self._display_counter,
        )
        self._rows[row_id] = row
        return row

    def _group_front_tangent(self, piece_ids: frozenset) -> float:
        tracks = [self._track_lookup[pid] for pid in piece_ids if pid in self._track_lookup]
        if not tracks:
            return 0.0
        return min(t.tangent_x for t in tracks)

    # ---------- geometry refresh ----------

    def _refresh_geometry(self) -> None:
        for row in self._rows.values():
            if row.state == RowGroupState.RETIRED:
                continue
            member_tracks = [
                self._track_lookup[pid] for pid in row.member_piece_ids if pid in self._track_lookup
            ]
            if not member_tracks:
                # All members currently untracked (dropped or, for a LOCKED
                # row, simply gone) — hold last known geometry, never crash.
                continue
            row.prev_front_tangent = row.front_tangent
            row.front_tangent = min(t.tangent_x for t in member_tracks)
            row.back_tangent = max(t.tangent_x for t in member_tracks)
            row.center_y_values = [t.center_y for t in member_tracks]

    # ---------- 2c. lifecycle ----------

    def _advance_lifecycle(self, frame_id: int, transfer_x: float) -> None:
        cfg = self._cfg
        for row in self._rows.values():
            if row.state == RowGroupState.RETIRED:
                continue

            if row.state == RowGroupState.CANDIDATE:
                self._log(
                    "ROW-STABILIZING", level="debug", frame=frame_id, row=row.row_id,
                    stable_updates=row.stable_updates, target=cfg.row_stabilize_frames,
                )
                if row.stable_updates >= cfg.row_stabilize_frames:
                    row.state = RowGroupState.STABILIZING

            elif row.state == RowGroupState.STABILIZING:
                self._log(
                    "ROW-STABILIZING", level="debug", frame=frame_id, row=row.row_id,
                    stable_updates=row.stable_updates, target=cfg.row_confirm_frames,
                )

            if row.state in (RowGroupState.CANDIDATE, RowGroupState.STABILIZING):
                members_ready = all(
                    self._track_lookup[pid].fresh_observations >= cfg.row_member_min_observations
                    for pid in row.member_piece_ids
                    if pid in self._track_lookup
                )
                if row.stable_updates >= cfg.row_confirm_frames and members_ready:
                    row.state = RowGroupState.CONFIRMED
                    row.confirmed_frame = frame_id
                    self._log(
                        "ROW-CONFIRMED", level="info", frame=frame_id, row=row.row_id,
                        disp=row.display_number, members=sorted(row.member_piece_ids),
                        front=row.front_tangent, back=row.back_tangent,
                        version=row.membership_version,
                    )

            if row.state == RowGroupState.CONFIRMED:
                if row.front_tangent <= transfer_x + cfg.row_lock_margin_px:
                    row.locked_frame = frame_id
                    row.locked_snapshot = LockedRowSnapshot(
                        cycle_id=row.cycle_id, row_id=row.row_id,
                        member_piece_ids=tuple(sorted(row.member_piece_ids)),
                        front_tangent=row.front_tangent, back_tangent=row.back_tangent,
                        piece_count=len(row.member_piece_ids),
                        membership_version=row.membership_version,
                        confirmed_frame=row.confirmed_frame, locked_frame=frame_id,
                    )
                    row.state = RowGroupState.LOCKED_FOR_TRANSFER
                    self._log(
                        "ROW-LOCKED", level="info", frame=frame_id, row=row.row_id,
                        disp=row.display_number, members=sorted(row.member_piece_ids),
                        front=row.front_tangent, back=row.back_tangent,
                        version=row.membership_version, transfer_x=transfer_x,
                    )

    # ---------- logging ----------

    def _log_evidence_reset(self, frame_id: int, row_id: int, kind: str, prev_count: int) -> None:
        if self._logger is None:
            return
        self._logger.debug(
            "[ROW] ROW-EVIDENCE-RESET frame={} row={} kind={} prev_count={}",
            frame_id, row_id, kind, prev_count,
        )

    def _log_summary(self, frame_id: int, raw_det_count: int | None, live_piece_count: int) -> None:
        if self._logger is None:
            return

        def _count(state: RowGroupState) -> int:
            return sum(1 for r in self._rows.values() if r.state == state)

        emitted = _count(RowGroupState.CONFIRMED) + _count(RowGroupState.LOCKED_FOR_TRANSFER)
        self._logger.info(
            "[ROW] ROW-SUMMARY frame={} raw_dets={} live_pieces={} candidate={} "
            "stabilizing={} confirmed={} locked={} emitted={}",
            frame_id, raw_det_count if raw_det_count is not None else "-", live_piece_count,
            _count(RowGroupState.CANDIDATE), _count(RowGroupState.STABILIZING),
            _count(RowGroupState.CONFIRMED), _count(RowGroupState.LOCKED_FOR_TRANSFER), emitted,
        )

    def _log(self, prefix: str, *, level: str, **kw) -> None:
        if self._logger is None:
            return

        if prefix == "ROW-GROUP-REJECT":
            msg = (
                "[ROW] ROW-GROUP-REJECT frame={} piece={} group_rows={} reason={} "
                "x_gap={:.1f} min_dy={:.1f} y_excl={:.1f}"
            )
            args = (kw["frame"], kw["piece"], kw["group_rows"], kw["reason"], kw["x_gap"], kw["min_dy"], kw["y_excl"])
        elif prefix == "ROW-CANDIDATE-CREATE":
            msg = "[ROW] ROW-CANDIDATE-CREATE frame={} row={} disp=R{} members={} front={:.1f} back={:.1f} ys={}"
            args = (kw["frame"], kw["row"], kw["disp"], kw["members"], kw["front"], kw["back"], kw["ys"])
        elif prefix == "ROW-MEMBERSHIP":
            msg = "[ROW] ROW-MEMBERSHIP frame={} row={} state={} members={} version={} result={} reason={}"
            args = (
                kw["frame"], kw["row"], kw["state"], kw["members"], kw["version"],
                kw["result"], kw["reason"],
            )
        elif prefix == "ROW-ASSIGNMENT-CHANGED":
            msg = (
                "[ROW] ROW-ASSIGNMENT-CHANGED frame={} piece={} prev_row={} proposed_row={} "
                "result_row={} evidence={}/{} score={:.2f} reason={}"
            )
            evidence = kw["evidence"]
            args = (
                kw["frame"], kw["piece"], kw["prev_row"], kw["proposed_row"], kw["result_row"],
                evidence[0], evidence[1], kw["score"], kw["reason"],
            )
        elif prefix == "ROW-STABILIZING":
            msg = "[ROW] ROW-STABILIZING frame={} row={} stable_updates={}/{}"
            args = (kw["frame"], kw["row"], kw["stable_updates"], kw["target"])
        elif prefix == "ROW-CONFIRMED":
            msg = "[ROW] ROW-CONFIRMED frame={} row={} disp=R{} members={} front={:.1f} back={:.1f} version={}"
            args = (kw["frame"], kw["row"], kw["disp"], kw["members"], kw["front"], kw["back"], kw["version"])
        elif prefix == "ROW-LOCKED":
            msg = (
                "[ROW] ROW-LOCKED frame={} row={} disp=R{} members={} front={:.1f} back={:.1f} "
                "version={} transfer_x={:.1f}"
            )
            args = (
                kw["frame"], kw["row"], kw["disp"], kw["members"], kw["front"], kw["back"],
                kw["version"], kw["transfer_x"],
            )
        elif prefix == "ROW-MERGE-PROPOSED":
            msg = "[ROW] ROW-MERGE-PROPOSED frame={} rows={} evidence={}/{} score_gain={:.2f}"
            evidence = kw["evidence"]
            args = (kw["frame"], kw["rows"], evidence[0], evidence[1], kw["score_gain"])
        elif prefix == "ROW-MERGE-ACCEPTED":
            msg = "[ROW] ROW-MERGE-ACCEPTED frame={} survivor={} retired={} members={} version={}"
            args = (kw["frame"], kw["survivor"], kw["retired"], kw["members"], kw["version"])
        elif prefix == "ROW-MERGE-REJECTED":
            msg = "[ROW] ROW-MERGE-REJECTED frame={} rows={} reason={}"
            args = (kw["frame"], kw["rows"], kw["reason"])
        elif prefix == "ROW-SPLIT-PROPOSED":
            msg = "[ROW] ROW-SPLIT-PROPOSED frame={} row={} groups={} evidence={}/{}"
            evidence = kw["evidence"]
            args = (kw["frame"], kw["row"], kw["groups"], evidence[0], evidence[1])
        elif prefix == "ROW-SPLIT-ACCEPTED":
            msg = "[ROW] ROW-SPLIT-ACCEPTED frame={} row={} kept_members={} new_row={} new_members={}"
            args = (kw["frame"], kw["row"], kw["kept_members"], kw["new_row"], kw["new_members"])
        elif prefix == "ROW-SPLIT-REJECTED":
            msg = "[ROW] ROW-SPLIT-REJECTED frame={} row={} reason={}"
            args = (kw["frame"], kw["row"], kw["reason"])
        else:  # pragma: no cover - defensive
            return
        getattr(self._logger, level)(msg, *args)
