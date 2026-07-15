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

from dataclasses import dataclass
from typing import NamedTuple, Sequence


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

    ``state`` is ``"PENDING"`` (not yet fired) or ``"DONE"`` (fired,
    permanent for the rest of the cycle — there is no third "clearing"
    state; once fired, a cluster is done).
    """

    id: int
    pieces: list[ClusteredPiece]
    state: str = "PENDING"
    missed_frames: int = 0
    fired_tangent_x: float | None = None

    @property
    def front_tangent_x(self) -> float:
        return min(p.tangent_x for p in self.pieces)


class ClusterTracker:
    """Tracks proximity clusters by identity across frames and drives the
    PENDING -> DONE (permanent) lifecycle behind cluster-based StopTuchabzug
    firing — see MainWindow._apply_cluster_stop_edge.

    Continuous by design: unlike the retired row/grid model there is no
    "reset to slot 0" — new clusters are picked up the moment they appear,
    with no limit on how many are tracked at once, and tracking survives
    TuchabzugRunning rising/falling edges (dough pieces don't disappear just
    because the signal toggles). Call reset() only on genuinely
    discontinuous events (detection disabled, wizard completion, cycle end).
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
        """Match this frame's clusters to tracked PENDING clusters by
        front_tangent_x proximity (greedy nearest-match within
        ``tolerance_px`` — the same knob already used for the clustering
        itself, not a separate tuning parameter).

        Unmatched current clusters become new PENDING tracked clusters (no
        limit on count — this is how new arrivals are picked up). Unmatched
        PENDING tracked clusters age by one missed frame and are dropped
        once ``missed_frames`` exceeds ``max_missed_frames`` (so a brief gap
        in detection, e.g. one dropped Hough frame, doesn't lose the
        cluster). DONE clusters are left untouched here: their own pieces
        are permanently excluded from the clustering input by
        DonePieceTracker, so they could never match a current cluster
        anyway — DONE is terminal, nothing transitions it further.
        """
        pending = [tc for tc in self.tracked if tc.state == "PENDING"]
        done = [tc for tc in self.tracked if tc.state != "PENDING"]
        matched_pending_ids: set[int] = set()
        matched_current_idx: set[int] = set()

        for ci, cur in enumerate(current_clusters):
            best_tc: TrackedCluster | None = None
            best_dist = tolerance_px
            for tc in pending:
                if tc.id in matched_pending_ids:
                    continue
                dist = abs(tc.front_tangent_x - cur.front_tangent_x)
                if dist <= best_dist:
                    best_dist = dist
                    best_tc = tc
            if best_tc is not None:
                best_tc.pieces = cur.pieces
                best_tc.missed_frames = 0
                matched_pending_ids.add(best_tc.id)
                matched_current_idx.add(ci)

        new_tracked: list[TrackedCluster] = []
        for tc in pending:
            if tc.id in matched_pending_ids:
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

        self.tracked = sorted(new_tracked + done, key=lambda tc: tc.front_tangent_x)
        return self.tracked

    def get_active_target(self) -> TrackedCluster | None:
        """The single cluster the fire decision watches this frame: among
        PENDING clusters, the one with the smallest front_tangent_x (closest
        to the transfer line). None if no PENDING clusters exist."""
        pending = [tc for tc in self.tracked if tc.state == "PENDING"]
        if not pending:
            return None
        return min(pending, key=lambda tc: tc.front_tangent_x)

    def mark_done(self, cluster_id: int, fired_tangent_x: float) -> TrackedCluster | None:
        """Flip a tracked cluster to DONE — permanent for the rest of the
        cycle. No CLEARING phase, no timeout: the cluster (and, via
        DonePieceTracker, its pieces) is simply never checked against
        transfer_x or eligible for re-clustering again until the next
        cycle."""
        for tc in self.tracked:
            if tc.id == cluster_id:
                tc.state = "DONE"
                tc.fired_tangent_x = fired_tangent_x
                return tc
        return None


@dataclass
class _DonePiece:
    """Last known position of one piece that belonged to a DONE cluster."""

    tangent_x: float
    center_y: float


class DonePieceTracker:
    """Tracks individual pieces that belonged to a DONE cluster, by
    identity, so they stay permanently excluded from forming or joining any
    new cluster for the rest of the cycle — even if they drift, or
    detection briefly loses and re-acquires them near the same spot.

    Uses the same nearest-match-within-tolerance approach ClusterTracker
    uses for whole clusters, applied per piece instead: each frame, a
    candidate piece within ``tolerance_px`` of a known done-piece's last
    position is recognized as that same piece (not by a static X/boundary
    cutoff) and its tracked position follows the match, so a done piece
    that drifts stays excluded rather than "escaping" a fixed line.
    """

    def __init__(self) -> None:
        self._pieces: list[_DonePiece] = []

    def reset(self) -> None:
        self._pieces = []

    def mark_done(self, pieces: Sequence[ClusteredPiece]) -> None:
        for p in pieces:
            self._pieces.append(_DonePiece(tangent_x=p.tangent_x, center_y=p.center_y))

    def filter_out_done(self, pieces: Sequence[Piece], tolerance_px: float) -> list[Piece]:
        """Return only the pieces that do NOT match a known done piece,
        updating matched done pieces' tracked position to follow drift."""
        if not self._pieces:
            return list(pieces)
        kept: list[Piece] = []
        for p in pieces:
            p_tangent_x = p.center_x - p.radius
            best_dp: _DonePiece | None = None
            best_dist = tolerance_px
            for dp in self._pieces:
                dist = ((p_tangent_x - dp.tangent_x) ** 2 + (p.center_y - dp.center_y) ** 2) ** 0.5
                if dist <= best_dist:
                    best_dist = dist
                    best_dp = dp
            if best_dp is not None:
                best_dp.tangent_x = p_tangent_x
                best_dp.center_y = p.center_y
            else:
                kept.append(p)
        return kept
