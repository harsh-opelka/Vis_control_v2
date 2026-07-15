"""Tangent-based proximity clustering, and cross-frame cluster tracking.

Groups detected dough pieces by proximity along ``tangent_x`` (a piece's
LEADING edge on the X axis: ``center_x - radius``), independent of Y
position. Pieces belonging to the same physical row can sit at very
different Y (column) positions but still cluster together here because only
tangent_x proximity is considered — this replaces fixed-grid/row assumptions
(grid size may vary or be unknown in the field) with a grid-free model.

A cluster is simply "pieces close enough in tangent_x to be transferred
together": no fixed expected count, no outlier rejection.

``cluster_by_tangent`` is a pure per-frame function (also used standalone by
MainWindow's diagnostic PROXIMITY-CLUSTER overlay/logging,
config.proximity_clustering). ``ClusterTracker`` builds on top of it to
track individual clusters by identity across frames and drive the actual
StopTuchabzug fire decision — see MainWindow._apply_cluster_stop_edge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import NamedTuple, Sequence

# Per-cluster CLEARING->dropped transition: consecutive frames with zero
# pieces behind the committed boundary required before a FIRED cluster is
# considered fully cleared. 1 = clear the instant the trailing pieces are
# gone (matches the old global boundary's _BOUNDARY_CLEAR_STREAK).
_CLUSTER_CLEAR_STREAK: int = 1


class Piece(NamedTuple):
    """A detected circle, in the coordinate space cluster_by_tangent needs."""

    center_x: float
    center_y: float
    radius: float


@dataclass
class ClusteredPiece:
    """One piece, augmented with its tangent_x, as returned inside a cluster."""

    tangent_x: float
    center_x: float
    center_y: float
    radius: float


@dataclass
class ProximityCluster:
    pieces: list[ClusteredPiece]

    @property
    def front_tangent_x(self) -> float:
        """The minimum tangent_x in this cluster — its "leading" piece."""
        return min(p.tangent_x for p in self.pieces)


def cluster_by_tangent(
    pieces: Sequence[Piece], tolerance_px: float,
) -> list[ProximityCluster]:
    """Group pieces by proximity along tangent_x = center_x - radius.

    Sorts pieces by tangent_x and starts a new cluster whenever the gap to
    the next piece exceeds ``tolerance_px``. No outlier rejection, no fixed
    expected cluster count/size — a cluster can hold 1 piece or many.
    """
    if not pieces:
        return []

    clustered = sorted(
        (
            ClusteredPiece(
                tangent_x=p.center_x - p.radius,
                center_x=p.center_x,
                center_y=p.center_y,
                radius=p.radius,
            )
            for p in pieces
        ),
        key=lambda cp: cp.tangent_x,
    )

    clusters: list[ProximityCluster] = [ProximityCluster(pieces=[clustered[0]])]
    for prev, cur in zip(clustered, clustered[1:]):
        if cur.tangent_x - prev.tangent_x > tolerance_px:
            clusters.append(ProximityCluster(pieces=[cur]))
        else:
            clusters[-1].pieces.append(cur)
    return clusters


def pieces_from_detections(detections: Sequence[object]) -> list[Piece]:
    """Adapt viscontrol ``Detection`` objects (``.centroid``/``.width_px``/
    ``.height_px``) into the plain ``(center_x, center_y, radius)`` shape
    :func:`cluster_by_tangent` works with."""
    out: list[Piece] = []
    for d in detections:
        cx, cy = d.centroid  # type: ignore[attr-defined]
        radius = max(float(d.width_px or 0.0), float(d.height_px or 0.0)) / 2.0  # type: ignore[attr-defined]
        out.append(Piece(center_x=float(cx), center_y=float(cy), radius=radius))
    return out


@dataclass
class TrackedCluster:
    """A proximity cluster tracked by identity across frames.

    ``state`` is ``"ACTIVE"`` (not yet fired) or ``"FIRED"`` (stopped,
    clearing its committed boundary) — a cluster that fully clears is
    dropped from tracking entirely rather than kept as a third state.
    """

    id: int
    pieces: list[ClusteredPiece]
    state: str = "ACTIVE"
    missed_frames: int = 0
    fired_tangent_x: float | None = None
    committed_boundary_x: float | None = None
    clearing_since: float | None = None
    clear_streak: int = 0

    @property
    def front_tangent_x(self) -> float:
        return min(p.tangent_x for p in self.pieces)


class ClusterTracker:
    """Tracks proximity clusters by identity across frames and drives the
    ACTIVE -> FIRED -> (cleared) lifecycle behind cluster-based
    StopTuchabzug firing — see MainWindow._apply_cluster_stop_edge.

    Continuous by design: unlike the retired row/grid model there is no
    "reset to slot 0" — new clusters are picked up the moment they appear,
    with no limit on how many are tracked at once, and tracking survives
    TuchabzugRunning rising/falling edges (dough pieces don't disappear just
    because the signal toggles). Call reset() only on genuinely
    discontinuous events (detection disabled, wizard completion).
    """

    def __init__(self) -> None:
        self.tracked: list[TrackedCluster] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracked = []
        self._next_id = 1

    def update(
        self,
        current_clusters: Sequence[ProximityCluster],
        tolerance_px: float,
        max_missed_frames: int,
    ) -> list[TrackedCluster]:
        """Match this frame's clusters to tracked ACTIVE clusters by
        front_tangent_x proximity (greedy nearest-match within
        ``tolerance_px`` — the same knob already used for the clustering
        itself, not a separate tuning parameter).

        Unmatched current clusters become new ACTIVE tracked clusters (no
        limit on count — this is how new arrivals are picked up). Unmatched
        ACTIVE tracked clusters age by one missed frame and are dropped once
        ``missed_frames`` exceeds ``max_missed_frames`` (so a brief gap in
        detection, e.g. one dropped Hough frame, doesn't lose the cluster).
        FIRED clusters are left untouched here: their own pieces are
        deliberately excluded from the clustering input while clearing (see
        the boundary filter in MainWindow), so they could never match a
        current cluster anyway — their lifecycle is driven by
        :meth:`evaluate_clearing` instead.
        """
        active = [tc for tc in self.tracked if tc.state == "ACTIVE"]
        fired = [tc for tc in self.tracked if tc.state != "ACTIVE"]
        matched_active_ids: set[int] = set()
        matched_current_idx: set[int] = set()

        for ci, cur in enumerate(current_clusters):
            best_tc: TrackedCluster | None = None
            best_dist = tolerance_px
            for tc in active:
                if tc.id in matched_active_ids:
                    continue
                dist = abs(tc.front_tangent_x - cur.front_tangent_x)
                if dist <= best_dist:
                    best_dist = dist
                    best_tc = tc
            if best_tc is not None:
                best_tc.pieces = cur.pieces
                best_tc.missed_frames = 0
                matched_active_ids.add(best_tc.id)
                matched_current_idx.add(ci)

        new_tracked: list[TrackedCluster] = []
        for tc in active:
            if tc.id in matched_active_ids:
                new_tracked.append(tc)
                continue
            tc.missed_frames += 1
            if tc.missed_frames <= max_missed_frames:
                new_tracked.append(tc)
            # else: dropped — left the ROI/view without firing.

        for ci, cur in enumerate(current_clusters):
            if ci in matched_current_idx:
                continue
            new_tracked.append(TrackedCluster(id=self._next_id, pieces=cur.pieces))
            self._next_id += 1

        self.tracked = sorted(new_tracked + fired, key=lambda tc: tc.front_tangent_x)
        return self.tracked

    def mark_fired(
        self, cluster_id: int, fired_tangent_x: float, boundary_offset_px: float,
    ) -> TrackedCluster | None:
        """Flip a tracked cluster to FIRED and start its own CLEARING clock
        immediately.

        No deferral to a "next rising edge" — the old row model deferred the
        clock start only because that edge was the row-slot-advance event;
        clusters have no such event to wait for, so starting the clock at
        fire time is a deliberate simplification.
        """
        for tc in self.tracked:
            if tc.id == cluster_id:
                tc.state = "FIRED"
                tc.fired_tangent_x = fired_tangent_x
                tc.committed_boundary_x = fired_tangent_x + boundary_offset_px
                tc.clearing_since = time.monotonic()
                tc.clear_streak = 0
                return tc
        return None

    def evaluate_clearing(
        self, all_tangent_xs: Sequence[float], clearing_timeout_s: float,
    ) -> list[TrackedCluster]:
        """Per-cluster port of the old global ARMED/CLEARING boundary logic.

        For each FIRED cluster, checks whether any piece — from the RAW,
        unfiltered detections, not the clustering-eligible subset — still
        sits at/behind its ``committed_boundary_x``. Zero such pieces for
        one frame, or ``clearing_timeout_s`` elapsed, clears the cluster
        (dropped from tracking). Returns the clusters cleared by this call,
        for the caller to log.
        """
        now = time.monotonic()
        cleared: list[TrackedCluster] = []
        remaining: list[TrackedCluster] = []
        for tc in self.tracked:
            if tc.state != "FIRED" or tc.committed_boundary_x is None:
                remaining.append(tc)
                continue
            excluded = [x for x in all_tangent_xs if x <= tc.committed_boundary_x]
            tc.clear_streak = 0 if excluded else tc.clear_streak + 1
            elapsed = now - (tc.clearing_since or now)
            if tc.clear_streak >= _CLUSTER_CLEAR_STREAK or elapsed >= clearing_timeout_s:
                cleared.append(tc)
            else:
                remaining.append(tc)
        self.tracked = remaining
        return cleared
