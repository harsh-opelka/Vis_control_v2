"""Tangent-based proximity clustering, and cross-frame cluster tracking.

Groups detected dough pieces by proximity along ``tangent_x`` (a piece's
LEADING edge on the X axis: ``center_x - radius``), independent of Y
position. Pieces belonging to the same physical row can sit at very
different Y (column) positions but still cluster together here because only
tangent_x proximity is considered — this replaces fixed-grid/row assumptions
(grid size may vary or be unknown in the field) with a grid-free model.

A cluster is simply "pieces close enough in tangent_x to be transferred
together": no fixed expected count, no outlier rejection.

``cluster_by_tangent`` is a pure per-frame function used both by
MainWindow's diagnostic PROXIMITY-CLUSTER overlay/logging
(config.proximity_clustering) and by the real StopTuchabzug fire path —
see viscontrol/core/transfer_orchestrator.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

from viscontrol.core.logger import logger


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


@dataclass
class _DonePiece:
    """Last known position of one piece that belonged to a DONE cluster."""

    tangent_x: float
    center_y: float


@dataclass
class UnconfirmedPiece:
    """A candidate piece that hasn't yet reached ``min_fresh_confirmations``.

    Same shape as :class:`Piece` plus ``fresh_count``, which only exists for
    logging/overlay purposes — never fed into ``cluster_by_tangent``.
    """

    center_x: float
    center_y: float
    radius: float
    fresh_count: int


@dataclass
class _ConfirmationCandidate:
    """One tracked candidate inside :class:`PieceConfirmationTracker`."""

    tangent_x: float
    center_x: float
    center_y: float
    radius: float
    fresh_count: int = 0
    last_fresh_frame: int = 0
    confirmed: bool = False


class PieceConfirmationTracker:
    """Requires a detection to appear in at least N consecutive FRESH Hough
    frames before it is eligible for clustering or firing.

    Only meant to be called once per fresh Hough result (a cached/stale
    frame must never be fed in — the caller enforces this by only calling
    ``update()`` from the Hough-gated fire path, see
    MainWindow._apply_cluster_stop_edge). Uses the same tangent_x-proximity
    matching to track candidate identity across
    frames. Pieces that fail confirmation are excluded from the eligible set
    passed to ``cluster_by_tangent`` in the real fire path, but are still
    returned (with their ``fresh_count``) for logging/overlay.
    """

    def __init__(self) -> None:
        self._candidates: list[_ConfirmationCandidate] = []

    def reset(self) -> None:
        self._candidates = []

    def update(
        self,
        pieces: Sequence[Piece],
        frame_count: int,
        tolerance_px: float,
        min_fresh_confirmations: int,
        max_memory_frames: int = 0,
    ) -> tuple[list[Piece], list[UnconfirmedPiece]]:
        """Match ``pieces`` (this frame's fresh detections) against tracked
        candidates by tangent_x proximity (within ``tolerance_px``),
        advance each match's ``fresh_count``, and split the result into
        pieces eligible for clustering (``fresh_count >=
        min_fresh_confirmations``) and pieces still awaiting confirmation.

        ``min_fresh_confirmations <= 1`` disables the gate entirely: every
        piece passes through unchanged and no candidate bookkeeping happens
        (zero overhead, matches existing behavior).
        """
        if min_fresh_confirmations <= 1:
            return list(pieces), []

        matched_idx: set[int] = set()
        confirmed: list[Piece] = []
        unconfirmed: list[UnconfirmedPiece] = []

        for piece in pieces:
            piece_tangent_x = piece.center_x - piece.radius
            best_idx: int | None = None
            best_dist = tolerance_px
            for idx, cand in enumerate(self._candidates):
                if idx in matched_idx:
                    continue
                dist = abs(cand.tangent_x - piece_tangent_x)
                if dist <= best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                cand = self._candidates[best_idx]
                matched_idx.add(best_idx)
            else:
                cand = _ConfirmationCandidate(
                    tangent_x=piece_tangent_x, center_x=piece.center_x,
                    center_y=piece.center_y, radius=piece.radius,
                )
                self._candidates.append(cand)
                matched_idx.add(len(self._candidates) - 1)

            was_confirmed = cand.confirmed
            cand.tangent_x = piece_tangent_x
            cand.center_x = piece.center_x
            cand.center_y = piece.center_y
            cand.radius = piece.radius
            cand.fresh_count += 1
            cand.last_fresh_frame = frame_count

            if cand.fresh_count >= min_fresh_confirmations:
                cand.confirmed = True
                confirmed.append(piece)
                if not was_confirmed:
                    logger.debug(
                        "[DET] CANDIDATE-CONFIRM tangent_x={} center_y={} "
                        "fresh_count={} frame={}",
                        round(cand.tangent_x, 1), round(cand.center_y, 1),
                        cand.fresh_count, frame_count,
                    )
            else:
                unconfirmed.append(UnconfirmedPiece(
                    center_x=piece.center_x, center_y=piece.center_y,
                    radius=piece.radius, fresh_count=cand.fresh_count,
                ))
                logger.info(
                    "[DET] CANDIDATE-UNCONFIRMED tangent_x={} center_y={} "
                    "fresh_count={} frame={}",
                    round(cand.tangent_x, 1), round(cand.center_y, 1),
                    cand.fresh_count, frame_count,
                )

        # Candidates not matched this frame are left untouched — their
        # fresh_count is never reset, only ever advanced on a real fresh
        # match (a cached/stale frame is never passed to update() at all).
        if max_memory_frames > 0:
            self._candidates = [
                c for c in self._candidates
                if (frame_count - c.last_fresh_frame) <= max_memory_frames
            ]

        return confirmed, unconfirmed
