"""Diagnostic-only transfer event tracker.

Tracks physical "transfer events" (a row/cluster of dough approaching the
transfer line) across frames, independent of the real cluster-identity
tracking that drives StopTuchabzug (see
``viscontrol/detection/proximity_clustering.py``: ``ClusterTracker``,
``MainWindow._apply_cluster_stop_edge``). This module never reads or writes
any state used by that real firing path — it only observes the same
per-frame ``cluster_by_tangent`` output and logs/visualizes what *would*
happen at the transfer line (``WOULD_FIRE``), for tuning and debugging.

Visualization and logging only. Do not wire this into any StopTuchabzug,
PLC, or fire-decision code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from viscontrol.core.logger import logger


@dataclass
class ClusterSnapshot:
    """One frame's proximity cluster, in the plain shape the tracker needs."""

    cluster_id: int          # current frame cluster id (unstable, for debug only)
    front_tangent: float     # min tangent_x of all pieces in cluster
    back_tangent: float      # max tangent_x of all pieces in cluster
    center_tangent: float    # midpoint of front and back
    piece_count: int
    center_y: float          # average y of circles in cluster (logged only, not used for matching)


@dataclass
class TransferEvent:
    event_id: int
    state: str                     # TRACKING | ARMED | WOULD_FIRE | DONE
    front_tangent: float           # leading edge (min tangent_x seen across all matched clusters this frame)
    back_tangent: float            # trailing edge (max tangent_x)
    center_tangent: float          # midpoint of front and back
    prev_front_tangent: float      # front_tangent from previous frame, for tripwire crossing detection
    first_seen_frame: int
    last_seen_frame: int
    observed_frames: int = 0       # number of frames this event has been seen (not counting missed frames)
    missed_frames: int = 0         # consecutive frames where no cluster matched this event
    split_merge_count: int = 0     # how many times multiple clusters matched this event in one frame
    would_fire_frame: int | None = None
    max_overshoot_px: float = 0.0  # how far past transfer_x the front was when WOULD_FIRE triggered
    piece_count: int = 0           # sum of piece_count across the source cluster(s) at creation time
    source_cluster_ids: list[int] = field(default_factory=list)  # cluster_id(s) grouped into this event at creation


def _range_distance(front_a: float, back_a: float, front_b: float, back_b: float) -> float:
    """Gap between two [front, back] ranges along tangent_x. 0 (or negative,
    clamped to 0) means overlap."""
    return max(0.0, front_a - back_b, front_b - back_a)


def _interval_distance(cluster: ClusterSnapshot, event: TransferEvent) -> float:
    """Gap between a cluster's [front, back] range and an event's [front, back]
    range along tangent_x. 0 (or negative, clamped to 0) means overlap."""
    return _range_distance(
        cluster.front_tangent, cluster.back_tangent,
        event.front_tangent, event.back_tangent,
    )


class TransferEventTracker:
    """Diagnostic-only cross-frame tracker for transfer events.

    Matches this frame's proximity clusters against previously-seen events by
    interval distance along tangent_x (a separate, looser identity model than
    ``ClusterTracker`` — this is for visualization/tripwire-timing analysis,
    not firing). See module docstring: never used by any real fire decision.
    """

    def __init__(
        self,
        *,
        event_match_distance_px: float = 120,
        event_new_row_min_upstream_px: float = 60,
        event_new_event_group_merge_px: float = 130,
        event_arm_min_frames: int = 5,
        event_max_missed_frames: int = 8,
        event_max_late_overshoot_px: float = 30,
    ) -> None:
        self.event_match_distance_px = event_match_distance_px
        self.event_new_row_min_upstream_px = event_new_row_min_upstream_px
        self.event_new_event_group_merge_px = event_new_event_group_merge_px
        self.event_arm_min_frames = event_arm_min_frames
        self.event_max_missed_frames = event_max_missed_frames
        self.event_max_late_overshoot_px = event_max_late_overshoot_px

        self.events: list[TransferEvent] = []
        self._next_id = 1

        # DIAGNOSTIC ONLY (Fix F, main_window.py FIRE-DECISION-CONFLICT
        # check): front_tangent values logged as EVENT-CREATE-SKIP-LATE in
        # the most recent update() call. Reset at the start of every
        # update(); never read by anything inside this class.
        self._last_skip_late_fronts: list[float] = []

        # EVENT-SUMMARY counters.
        self._total_created = 0
        self._total_events_single = 0
        self._total_events_grouped = 0
        self._total_would_fire = 0
        self._total_skip_late = 0
        self._total_ignore_late = 0
        self._total_ambiguous_match = 0
        self._total_split_merge = 0
        self._max_missed_seen = 0
        self._max_overshoot_seen = 0.0

    def update(
        self,
        clusters: list[ClusterSnapshot],
        current_frame: int,
        transfer_x: float,
    ) -> None:
        """Match this frame's clusters against tracked events, update event
        positions/state, and log everything. See module docstring."""
        # Reset the SKIP-LATE front list at the start of every call — see
        # get_last_skip_late_fronts().
        self._last_skip_late_fronts = []

        # Step 1: tentatively age every existing event by one missed frame.
        # Snapshot the pre-increment value for EVENT-MATCH's missed_before.
        missed_before_by_id = {ev.event_id: ev.missed_frames for ev in self.events}
        for ev in self.events:
            ev.missed_frames += 1

        # Step 2/3: for each cluster, find candidate events within the match
        # distance and choose the closest one.
        chosen_event_for_cluster: dict[int, TransferEvent] = {}
        for ci, cluster in enumerate(clusters):
            candidates = [
                (ev, _interval_distance(cluster, ev))
                for ev in self.events
                if _interval_distance(cluster, ev) < self.event_match_distance_px
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda pair: pair[1])
            best_ev, _best_dist = candidates[0]
            chosen_event_for_cluster[ci] = best_ev
            if len(candidates) > 1:
                self._total_ambiguous_match += 1
                logger.info(
                    "[TRACKER] EVENT-AMBIGUOUS-MATCH cluster={} candidates={} "
                    "chosen={} distances={} frame={}",
                    cluster.cluster_id,
                    [ev.event_id for ev, _ in candidates],
                    best_ev.event_id,
                    [round(d, 1) for _, d in candidates],
                    current_frame,
                )

        # Step 4/5: group clusters by chosen event (split-merge), then apply
        # the match — update positions, reset missed_frames, log.
        clusters_by_event_id: dict[int, list[ClusterSnapshot]] = {}
        for ci, ev in chosen_event_for_cluster.items():
            clusters_by_event_id.setdefault(ev.event_id, []).append(clusters[ci])

        for ev in self.events:
            matched_clusters = clusters_by_event_id.get(ev.event_id)
            if not matched_clusters:
                continue

            new_front = min(c.front_tangent for c in matched_clusters)
            new_back = max(c.back_tangent for c in matched_clusters)
            new_center = (new_front + new_back) / 2.0

            if len(matched_clusters) > 1:
                ev.split_merge_count += 1
                self._total_split_merge += 1
                logger.info(
                    "[TRACKER] EVENT-SPLIT-MERGE id={} clusters={} front={} back={} frame={}",
                    ev.event_id,
                    [c.cluster_id for c in matched_clusters],
                    round(new_front, 1),
                    round(new_back, 1),
                    current_frame,
                )

            missed_before = missed_before_by_id.get(ev.event_id, 0)
            ev.prev_front_tangent = ev.front_tangent
            ev.front_tangent = new_front
            ev.back_tangent = new_back
            ev.center_tangent = new_center
            ev.missed_frames = 0
            ev.observed_frames += 1
            ev.last_seen_frame = current_frame

            for c in matched_clusters:
                logger.info(
                    "[TRACKER] EVENT-MATCH id={} cluster_id={} front={} back={} "
                    "missed_before={} frame={}",
                    ev.event_id, c.cluster_id, round(new_front, 1), round(new_back, 1),
                    missed_before, current_frame,
                )

        # Step 6: group unmatched upstream clusters into candidate physical
        # rows, then create one TransferEvent per candidate group. This
        # keeps e.g. two side-by-side pieces of the same row from becoming
        # two separate events. Unmatched clusters that aren't safely
        # upstream of the transfer line yet are skipped instead of grouped
        # (an existing event may still pick them up next frame; if none
        # ever does, they simply age out with the frame).
        upstream_unmatched: list[ClusterSnapshot] = []
        for ci, cluster in enumerate(clusters):
            if ci in chosen_event_for_cluster:
                continue
            if cluster.front_tangent > transfer_x + self.event_new_row_min_upstream_px:
                upstream_unmatched.append(cluster)
            else:
                self._total_skip_late += 1
                self._last_skip_late_fronts.append(cluster.front_tangent)
                logger.info(
                    "[TRACKER] EVENT-CREATE-SKIP-LATE cluster_id={} front={} transfer_x={} frame={}",
                    cluster.cluster_id, round(cluster.front_tangent, 1),
                    round(transfer_x, 1), current_frame,
                )

        # Farthest upstream first (largest tangent_x first): pieces move
        # from high tangent_x toward low tangent_x / toward the transfer
        # line, so this orders clusters from oldest-upstream to newest.
        upstream_unmatched.sort(key=lambda c: c.front_tangent, reverse=True)

        groups: list[list[ClusterSnapshot]] = []
        for cluster in upstream_unmatched:
            if groups:
                group_front = min(c.front_tangent for c in groups[-1])
                group_back = max(c.back_tangent for c in groups[-1])
                if _range_distance(
                    cluster.front_tangent, cluster.back_tangent, group_front, group_back
                ) < self.event_new_event_group_merge_px:
                    groups[-1].append(cluster)
                    continue
            groups.append([cluster])

        for group in groups:
            sorted_group = sorted(group, key=lambda c: c.front_tangent, reverse=True)
            fronts = [c.front_tangent for c in sorted_group]
            backs = [c.back_tangent for c in sorted_group]
            source_ids = [c.cluster_id for c in sorted_group]
            group_front = min(fronts)
            group_back = max(backs)
            group_center = (group_front + group_back) / 2.0
            piece_count = sum(c.piece_count for c in sorted_group)

            if len(sorted_group) > 1:
                max_gap = max(
                    _range_distance(
                        sorted_group[i].front_tangent, sorted_group[i].back_tangent,
                        sorted_group[i + 1].front_tangent, sorted_group[i + 1].back_tangent,
                    )
                    for i in range(len(sorted_group) - 1)
                )
            else:
                max_gap = 0.0

            logger.info(
                "[TRACKER] EVENT-CANDIDATE-GROUP clusters={} fronts={} backs={} "
                "group_front={} group_back={} gap={} frame={}",
                source_ids,
                [round(f, 1) for f in fronts],
                [round(b, 1) for b in backs],
                round(group_front, 1), round(group_back, 1), round(max_gap, 1),
                current_frame,
            )

            new_event = TransferEvent(
                event_id=self._next_id,
                state="TRACKING",
                front_tangent=group_front,
                back_tangent=group_back,
                center_tangent=group_center,
                prev_front_tangent=group_front,
                first_seen_frame=current_frame,
                last_seen_frame=current_frame,
                observed_frames=1,
                missed_frames=0,
                piece_count=piece_count,
                source_cluster_ids=source_ids,
            )
            self._next_id += 1
            self.events.append(new_event)
            self._total_created += 1
            grouped = len(sorted_group) > 1
            if grouped:
                self._total_events_grouped += 1
            else:
                self._total_events_single += 1
            logger.info(
                "[TRACKER] EVENT-CREATE id={} source_clusters={} front={} back={} "
                "piece_count={} grouped={} frame={}",
                new_event.event_id, source_ids,
                round(group_front, 1), round(group_back, 1), piece_count,
                "yes" if grouped else "no", current_frame,
            )

        # Step 7: log a MISS for every event with a nonzero miss streak this
        # frame (excludes events matched above, and events just created this
        # frame, both of which have missed_frames == 0).
        for ev in self.events:
            if ev.missed_frames == 0:
                continue
            logger.info(
                "[TRACKER] EVENT-MISS id={} missed={} frame={}",
                ev.event_id, ev.missed_frames, current_frame,
            )

        # Step 8: drop events that have been missed too many frames in a row.
        self._max_missed_seen = max(
            [self._max_missed_seen] + [ev.missed_frames for ev in self.events]
        )
        survivors: list[TransferEvent] = []
        for ev in self.events:
            if ev.missed_frames >= self.event_max_missed_frames:
                logger.info("[TRACKER] EVENT-DROP id={} frame={}", ev.event_id, current_frame)
            else:
                survivors.append(ev)
        self.events = survivors

        # State transitions, per event, after this frame's matching settles.
        for ev in self.events:
            old_state = ev.state
            if ev.state == "WOULD_FIRE":
                ev.state = "DONE"
                logger.info(
                    "[TRACKER] EVENT-STATE id={} {}->{} frame={}",
                    ev.event_id, old_state, ev.state, current_frame,
                )
            elif ev.state == "ARMED":
                crossed = ev.prev_front_tangent > transfer_x >= ev.front_tangent
                overshoot = transfer_x - ev.front_tangent
                logger.info(
                    "[TRACKER] EVENT-TRIPWIRE id={} prev_front={} cur_front={} "
                    "transfer_x={} crossed={} overshoot={} frame={}",
                    ev.event_id, round(ev.prev_front_tangent, 1), round(ev.front_tangent, 1),
                    round(transfer_x, 1), "yes" if crossed else "no",
                    round(overshoot, 1), current_frame,
                )
                if crossed:
                    ev.state = "WOULD_FIRE"
                    ev.would_fire_frame = current_frame
                    ev.max_overshoot_px = overshoot
                    self._total_would_fire += 1
                    self._max_overshoot_seen = max(self._max_overshoot_seen, overshoot)
                    logger.info(
                        "[TRACKER] EVENT-STATE id={} {}->{} frame={}",
                        ev.event_id, old_state, ev.state, current_frame,
                    )
            elif ev.state == "TRACKING":
                if (
                    ev.observed_frames >= self.event_arm_min_frames
                    and ev.front_tangent > transfer_x
                ):
                    ev.state = "ARMED"
                    logger.info(
                        "[TRACKER] EVENT-STATE id={} {}->{} frame={}",
                        ev.event_id, old_state, ev.state, current_frame,
                    )

    def get_last_skip_late_fronts(self) -> list[float]:
        """DIAGNOSTIC ONLY (Fix F): front_tangent values of every cluster
        logged as EVENT-CREATE-SKIP-LATE in the most recent update() call.
        Used only by main_window.py's FIRE-DECISION-CONFLICT check — never
        read by anything that affects firing."""
        return list(self._last_skip_late_fronts)

    def shutdown(self) -> None:
        """Log a final EVENT-SUMMARY line. Call once when the tracker (or the
        application) is shutting down."""
        ignored_late_count = self._total_skip_late + self._total_ignore_late
        logger.info(
            "[TRACKER] EVENT-SUMMARY total_events_created={} events_from_single_cluster={} "
            "events_from_grouped_clusters={} would_fire_count={} ignored_late_count={} "
            "ambiguous_match_count={} split_merge_count={} max_missed_frames_seen={} "
            "max_overshoot_px={}",
            self._total_created, self._total_events_single, self._total_events_grouped,
            self._total_would_fire, ignored_late_count, self._total_ambiguous_match,
            self._total_split_merge, self._max_missed_seen, round(self._max_overshoot_seen, 1),
        )
