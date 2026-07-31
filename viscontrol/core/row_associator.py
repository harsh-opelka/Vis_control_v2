"""Per-frame association of cluster observations to existing rows.

This replaces viscontrol/transfer_event_tracker.py. It does association
ONLY: it owns no transfer state, issues no commands, and knows nothing about
the PLC. See transfer_orchestrator.py for the lifecycle/fire logic that
consumes AssociationResult.

The matching/grouping algorithm below is ported exactly from the retired
TransferEventTracker (interval distance, upstream filter, greedy
group-merge sweep sorted farthest-upstream-first) — see refactor_inventory.txt.

No Qt imports. No OpenCV imports.
"""

from __future__ import annotations

from dataclasses import dataclass

from viscontrol.core.transfer_events import ClusterObservation, RowRecord, RowState

_TERMINAL_STATES = (RowState.TRANSFERRED, RowState.ABANDONED)


@dataclass
class AssociationResult:
    matches: dict  # {row_id: (front_tangent, back_tangent, piece_count)}
    new_row_candidates: list  # [(front, back, piece_count, source_indices)]
    skipped_late: list  # [(front, back, reason)]
    ambiguous: list  # [(obs_index, [candidate_row_ids], [distances], chosen)]
    missed_row_ids: list  # rows with no matching observation this frame


def _row_interval_distance(obs: ClusterObservation, row: RowRecord) -> float:
    """X-only interval distance between an observation's [front, back] range
    and a row's [front, back] range. 0 (or negative, clamped) = overlap.
    center_y is NEVER used here or anywhere else in this file."""
    return max(
        0.0,
        obs.front_tangent - row.back_tangent,
        row.front_tangent - obs.back_tangent,
    )


def _range_interval_distance(front_a: float, back_a: float, front_b: float, back_b: float) -> float:
    return max(0.0, front_a - back_b, front_b - back_a)


class RowAssociator:
    """Matches this frame's cluster observations against existing rows by
    tangent_x interval distance, and proposes new-row candidates for
    unmatched observations.

    Never mutates a RowRecord and never decides a state transition — pure
    read + report. ``cfg`` needs: row_match_distance_px, row_group_merge_px,
    row_new_min_upstream_px, row_new_min_gap_px.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def reset(self) -> None:
        """No internal state to reset (association is a pure per-call
        computation over the existing_rows passed to associate()); kept for
        interface symmetry with the tracker this replaces, called on
        CYCLE_START."""

    def associate(
        self,
        observations: list[ClusterObservation],
        frame_id: int,
        transfer_x: float,
        existing_rows: list[RowRecord],
    ) -> AssociationResult:
        candidate_rows = [r for r in existing_rows if r.state not in _TERMINAL_STATES]

        # 2. MATCHING: observation -> existing row, smallest interval
        # distance wins; multiple candidates within tolerance are recorded
        # as ambiguous but never merged.
        chosen_row_for_obs: dict[int, RowRecord] = {}
        ambiguous: list = []
        for oi, obs in enumerate(observations):
            candidates = [
                (row, _row_interval_distance(obs, row))
                for row in candidate_rows
                if _row_interval_distance(obs, row) < self._cfg.row_match_distance_px
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda pair: pair[1])
            best_row, _best_dist = candidates[0]
            chosen_row_for_obs[oi] = best_row
            if len(candidates) > 1:
                ambiguous.append((
                    oi,
                    [row.row_id for row, _ in candidates],
                    [round(d, 1) for _, d in candidates],
                    best_row.row_id,
                ))

        # Multiple observations may legally match the SAME row (split case)
        # — merge them into one match entry.
        obs_by_row_id: dict[int, list[ClusterObservation]] = {}
        for oi, row in chosen_row_for_obs.items():
            obs_by_row_id.setdefault(row.row_id, []).append(observations[oi])

        matches: dict = {}
        for row_id, matched_obs in obs_by_row_id.items():
            front = min(o.front_tangent for o in matched_obs)
            back = max(o.back_tangent for o in matched_obs)
            piece_count = sum(o.piece_count for o in matched_obs)
            matches[row_id] = (front, back, piece_count)

        # 3a/3b. NEW ROW CANDIDATES: reject unmatched observations that
        # aren't safely upstream, or that sit too close to an existing
        # non-terminal row (this is what prevents Row 3 being absorbed into
        # Row 2 and vice versa).
        skipped_late: list = []
        upstream_survivors: list[ClusterObservation] = []
        for oi, obs in enumerate(observations):
            if oi in chosen_row_for_obs:
                continue
            if obs.front_tangent <= transfer_x + self._cfg.row_new_min_upstream_px:
                skipped_late.append((obs.front_tangent, obs.back_tangent, "not_upstream"))
                continue
            too_close_row_id = None
            best_gap = self._cfg.row_new_min_gap_px
            for row in candidate_rows:
                d = _row_interval_distance(obs, row)
                if d < best_gap:
                    best_gap = d
                    too_close_row_id = row.row_id
            if too_close_row_id is not None:
                skipped_late.append(
                    (obs.front_tangent, obs.back_tangent, f"too_close_to_row_{too_close_row_id}")
                )
                continue
            upstream_survivors.append(obs)

        # 3c. GROUP the survivors before creating candidates: farthest
        # upstream first, greedy sweep merging by interval distance to the
        # current group's range. One physical multi-column row must produce
        # ONE candidate, not one per observation.
        upstream_survivors.sort(key=lambda o: o.front_tangent, reverse=True)
        groups: list[list[ClusterObservation]] = []
        for obs in upstream_survivors:
            if groups:
                g_front = min(o.front_tangent for o in groups[-1])
                g_back = max(o.back_tangent for o in groups[-1])
                if _range_interval_distance(
                    obs.front_tangent, obs.back_tangent, g_front, g_back
                ) < self._cfg.row_group_merge_px:
                    groups[-1].append(obs)
                    continue
            groups.append([obs])

        new_row_candidates: list = []
        for group in groups:
            front = min(o.front_tangent for o in group)
            back = max(o.back_tangent for o in group)
            piece_count = sum(o.piece_count for o in group)
            source_indices = [o.temp_cluster_index for o in group]
            new_row_candidates.append((front, back, piece_count, source_indices))

        # 4. MISSED ROWS: any non-terminal row with no matched observation.
        matched_row_ids = set(matches.keys())
        missed_row_ids = [r.row_id for r in candidate_rows if r.row_id not in matched_row_ids]

        return AssociationResult(
            matches=matches,
            new_row_candidates=new_row_candidates,
            skipped_late=skipped_late,
            ambiguous=ambiguous,
            missed_row_ids=missed_row_ids,
        )
