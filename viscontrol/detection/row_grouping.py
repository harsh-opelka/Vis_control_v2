"""Row grouping for the cloth side (behind the USE_ROW_GROUPING toggle).

Background
----------
The legacy cloth tripwire (see MainWindow._apply_tripwire_edge) watches a single
vertical strip at the transfer line and fires StopTuchabzug on an occupied→clear
edge. That fires ONCE per "occupied" episode, so two rows of dough that arrive
back-to-back with no clear gap between them look like a single long occupancy and
trigger only one stop.

Row grouping fixes that without touching detection or the tripwire. It works on
the centroids the active detection method already produces inside the cloth ROI:

1. ``group_into_rows`` clusters those centroids by their travel-direction
   coordinate (x — the transfer line is vertical, so the cloth travels along x).
   Pieces within a tolerance along travel — gap_diameters × the median DETECTED
   piece diameter — belong to the same row; a clearly larger jump starts the
   next row. The tolerance scales with dough SIZE (no fixed pixel sizes, no fixed
   row count), so slightly staggered/irregular pieces merge into one row instead
   of over-segmenting.
2. Each row collapses to a single "row-line": the MEDIAN travel coordinate of
   its members (robust to an outlier piece).
3. ``RowLineTracker`` follows those row-lines frame to frame and fires exactly
   once per row, the moment a tracked row-line crosses the transfer line.

Two rows that touch (no gap) still sit at two distinct travel positions, so they
form two clusters → two row-lines → two independent stops. That is the whole
point of the feature.

Nothing here sends the PLC pulse or changes any signal — it only decides *how
many* rows just crossed. MainWindow turns that count into the existing
StopTuchabzug pulse(s).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from viscontrol.core.logger import logger

# Row split tolerance, as a multiple of the median DETECTED piece diameter:
# pieces whose travel coordinates differ by less than
# (gap_diameters × median diameter) are the same row; a clearly larger gap
# starts the next row. Tolerance scales with dough size (no fixed pixels) and is
# overridable per call from config.detection.row_grouping_gap_diameters.
DEFAULT_GAP_DIAMETERS: float = 0.6

# Float-noise floor so the split threshold stays strictly positive even if the
# piece diameter can't be measured. Not a row size.
_GAP_EPS: float = 1e-3

# Per-frame matching: a current row-line is the same tracked row as a previous
# one when within MATCH_TOL_FACTOR × the median row spacing. The cloth moves only
# a little per frame (motion ≪ row spacing), so a fraction keeps a row from being
# matched to its neighbour.
MATCH_TOL_FACTOR: float = 0.4

# Drop a tracked row that hasn't been seen for this many consecutive frames
# (it has left the field of view / detection dropped it for good).
MAX_MISSED_FRAMES: int = 5


def median_piece_diameter(detections: Sequence[Any]) -> float:
    """Median detected piece diameter (mean of width_px/height_px per piece).

    Returns 0.0 when no detection carries a usable size — the caller then
    substitutes a fallback (e.g. the profile's expected_width_px).
    """
    diams: list[float] = []
    for d in detections:
        w = float(getattr(d, "width_px", 0.0) or 0.0)
        h = float(getattr(d, "height_px", 0.0) or 0.0)
        if w > 0.0 and h > 0.0:
            diams.append((w + h) / 2.0)
        elif w > 0.0 or h > 0.0:
            diams.append(max(w, h))
    if not diams:
        return 0.0
    return statistics.median(diams)


def leading_edge_x(detection: Any) -> float:
    """SECTION 5: travel-axis coordinate of a piece's LEADING edge.

    The transfer line is on the LEFT and the cloth moves left, so the leading
    (front) edge facing the line is the leftmost point of the circle:
    ``centroid_x - radius``. Using this instead of the centroid makes a stop
    fire when the FRONT of the piece arrives at the transfer point, which is
    physically correct (the centre is half a piece too late).
    """
    cx = float(detection.centroid[0])
    w = float(getattr(detection, "width_px", 0.0) or 0.0)
    h = float(getattr(detection, "height_px", 0.0) or 0.0)
    diam = (w + h) / 2.0 if (w > 0 and h > 0) else max(w, h)
    return cx - diam / 2.0


def group_rows(
    detections: Sequence[Any],
    *,
    gap_diameters: float = DEFAULT_GAP_DIAMETERS,
    piece_diameter: float | None = None,
) -> list[list[Any]]:
    """Cluster detections into rows by LEADING-EDGE travel coordinate.

    Like :func:`group_into_rows` but returns the actual member detections per
    row (not just a representative coordinate) and clusters on the leading edge
    (SECTION 5) rather than the centroid. Rows are returned front-first (the
    row whose leading edge is closest to the transfer line first). The split
    tolerance is still ``gap_diameters × piece_diameter`` — relative to dough
    size, never a fixed pixel count.
    """
    items = list(detections)
    if not items:
        return []
    items.sort(key=leading_edge_x)
    if piece_diameter is None:
        piece_diameter = median_piece_diameter(items)
    threshold = max(piece_diameter * gap_diameters, _GAP_EPS)

    rows: list[list[Any]] = []
    cluster: list[Any] = [items[0]]
    for prev, cur in zip(items, items[1:]):
        if leading_edge_x(cur) - leading_edge_x(prev) > threshold:
            rows.append(cluster)
            cluster = [cur]
        else:
            cluster.append(cur)
    rows.append(cluster)
    return rows


def group_by_gap(
    detections: Sequence[Any],
    gap_threshold_px: float,
) -> tuple[list[list[Any]], list[float]]:
    """Sort detections by leading_edge_x and cluster on the gaps between them.

    A new cluster starts whenever the gap between consecutive sorted pieces'
    leading-edge X exceeds ``gap_threshold_px``. Each cluster lines up with a
    physical row regardless of how many pieces were actually detected in it —
    unlike a fixed-size slice, a missing or extra detection in one row cannot
    shift the grouping of every row behind it.

    Returns ``(clusters, gaps)`` where ``clusters`` is front-first (smallest
    leading_edge_x first) and ``gaps`` is the consecutive-gap list used to
    split them, for the caller to log. Returns ``([], [])`` for no detections.
    ``gap_threshold_px`` <= 0 puts every piece in its own cluster.
    """
    items = list(detections)
    if not items:
        return [], []
    items.sort(key=leading_edge_x)
    clusters: list[list[Any]] = [[items[0]]]
    gaps: list[float] = []
    for prev, cur in zip(items, items[1:]):
        gap = leading_edge_x(cur) - leading_edge_x(prev)
        gaps.append(gap)
        if gap > gap_threshold_px:
            clusters.append([cur])
        else:
            clusters[-1].append(cur)
    return clusters, gaps


# Sticky cluster tracking (Layer 2, on top of group_by_gap): group_by_gap
# re-clusters from scratch every frame, so a cluster's position in the
# returned list is not a stable identity — a piece entering or leaving a row
# can shift every cluster behind it by one slot. ClusterTracker assigns each
# cluster a persistent id by matching it, frame to frame, to the tracked
# cluster whose PREDICTED front position (extrapolated by cloth speed) is
# nearest — never by list index — so callers can key row identity, UI color,
# and anchor selection off that id instead of per-frame sort order.
CLUSTER_MAX_MATCH_DIST_PX: float = 120.0
CLUSTER_MAX_MISSED_FRAMES: int = 3
DEFAULT_CLOTH_SPEED_PX_S: float = 350.0


@dataclass
class _TrackedCluster:
    id: int
    front_x: float         # last known front (min leading_edge_x of members)
    last_seen_frame: int
    missed: int = 0


class ClusterTracker:
    """Sticky per-cycle identity tracking for ``group_by_gap`` output.

    ``update`` is fed one frame's clusters and returns them tagged with a
    STABLE id. A cluster keeps its id across frames by matching its front
    position (min leading_edge_x of its members) to the nearest tracked
    cluster's PREDICTED front position — last front-X minus expected cloth
    travel since the last frame (``cloth_speed_px_s * dt_s``) — using a
    global nearest-pair match so a cluster can't steal a better-matching
    neighbour's id. A cluster farther than ``max_match_dist_px`` from every
    prediction is new (entering from the right, gets a fresh id). A tracked
    cluster unmatched for more than ``max_missed_frames`` frames is dropped
    (crossed the transfer line or lost for good).
    """

    def __init__(
        self,
        *,
        max_match_dist_px: float = CLUSTER_MAX_MATCH_DIST_PX,
        max_missed_frames: int = CLUSTER_MAX_MISSED_FRAMES,
    ) -> None:
        self._tracked: list[_TrackedCluster] = []
        self._next_id: int = 1
        self._frame_no: int = 0
        self._max_match_dist_px = max_match_dist_px
        self._max_missed_frames = max_missed_frames

    def reset(self) -> None:
        self._tracked = []
        self._next_id = 1
        self._frame_no = 0

    def update(
        self,
        clusters: Sequence[list[Any]],
        *,
        dt_s: float,
        cloth_speed_px_s: float = DEFAULT_CLOTH_SPEED_PX_S,
    ) -> list[tuple[int, list[Any]]]:
        """Tag this frame's clusters with stable ids.

        Returns ``(id, members)`` pairs in the SAME order as ``clusters``
        (front-first, per ``group_by_gap``). Logs one
        ``CLUSTER-TRACK: id=N front=X matched|new|lost`` line per tracked
        cluster this call, for visibility into row identity across frames.
        """
        self._frame_no += 1
        travel = max(cloth_speed_px_s, 0.0) * max(dt_s, 0.0)
        fronts = [
            min(leading_edge_x(d) for d in c) if c else float("inf")
            for c in clusters
        ]

        # Global nearest-pair greedy match: closest (cluster, tracked) pairs
        # win first, so a cluster can't be stolen by a farther-but-earlier one.
        candidates: list[tuple[float, int, int]] = []
        for ci, front in enumerate(fronts):
            for ti, tr in enumerate(self._tracked):
                predicted = tr.front_x - travel
                dist = abs(front - predicted)
                if dist <= self._max_match_dist_px:
                    candidates.append((dist, ci, ti))
        candidates.sort(key=lambda c: c[0])

        assigned_ids: list[int | None] = [None] * len(clusters)
        matched_tracked: set[int] = set()
        matched_clusters: set[int] = set()
        for _dist, ci, ti in candidates:
            if ci in matched_clusters or ti in matched_tracked:
                continue
            matched_clusters.add(ci)
            matched_tracked.add(ti)
            tr = self._tracked[ti]
            tr.front_x = fronts[ci]
            tr.last_seen_frame = self._frame_no
            tr.missed = 0
            assigned_ids[ci] = tr.id
            logger.info("CLUSTER-TRACK: id={} front={} matched", tr.id, int(round(fronts[ci])))

        for ci, front in enumerate(fronts):
            if assigned_ids[ci] is not None:
                continue
            tr = _TrackedCluster(id=self._next_id, front_x=front, last_seen_frame=self._frame_no)
            self._next_id += 1
            self._tracked.append(tr)
            assigned_ids[ci] = tr.id
            logger.info("CLUSTER-TRACK: id={} front={} new", tr.id, int(round(front)))

        for ti, tr in enumerate(self._tracked):
            if ti not in matched_tracked and tr.last_seen_frame != self._frame_no:
                tr.missed += 1
        lost = [tr for tr in self._tracked if tr.missed > self._max_missed_frames]
        for tr in lost:
            logger.info("CLUSTER-TRACK: id={} front={} lost", tr.id, int(round(tr.front_x)))
        self._tracked = [tr for tr in self._tracked if tr.missed <= self._max_missed_frames]

        return [(assigned_ids[i], clusters[i]) for i in range(len(clusters))]  # type: ignore[misc]


# Grouping outlier rejection: within a sliced group, a member whose leading-edge
# X differs from the median of the OTHER members by more than this many pixels
# is treated as a straggler (e.g. a leftover piece that slipped past the
# boundary filter) rather than a genuine member of that row.
GROUP_OUTLIER_MAX_DIST_PX: float = 100.0


def reject_group_outliers(
    group: Sequence[Any],
    *,
    max_dist_px: float = GROUP_OUTLIER_MAX_DIST_PX,
) -> tuple[list[Any], list[tuple[Any, float, float]]]:
    """Leave-one-out outlier rejection within a single sliced row group.

    For each member, compares its leading-edge X to the MEDIAN leading-edge X
    of the OTHER members. A member farther than ``max_dist_px`` from that
    median doesn't belong with the rest of the group and is dropped. Needs at
    least 2 members to have any "others" to compare against.

    Returns ``(kept, rejected)`` where ``rejected`` is
    ``(detection, median_of_others, distance)`` per dropped piece, for the
    caller to log.
    """
    items = list(group)
    if len(items) < 2:
        return items, []
    kept: list[Any] = []
    rejected: list[tuple[Any, float, float]] = []
    for i, d in enumerate(items):
        others = [leading_edge_x(o) for j, o in enumerate(items) if j != i]
        med = statistics.median(others)
        dist = abs(leading_edge_x(d) - med)
        if dist > max_dist_px:
            rejected.append((d, med, dist))
        else:
            kept.append(d)
    return kept, rejected


def row_leading_edge(row: Sequence[Any]) -> float:
    """Representative leading-edge travel position of a row (median of members,
    robust to one outlier piece)."""
    if not row:
        return 0.0
    return statistics.median(leading_edge_x(d) for d in row)


def front_row_by_grid(detections: Sequence[Any], columns: int) -> list[Any]:
    """SECTION 6: the FRONT/CURRENT row as the ``columns`` pieces closest to
    the transfer line — i.e. with the smallest leading-edge travel coordinate
    (the line is on the left). Returns all detections when there are fewer than
    ``columns`` of them, or ``columns <= 0``. Uses relative leading-edge
    positions only (no fixed pixel sizes)."""
    items = sorted(detections, key=leading_edge_x)
    if columns <= 0 or len(items) <= columns:
        return items
    return items[:columns]


def group_into_rows(
    detections: Sequence[Any],
    *,
    gap_diameters: float = DEFAULT_GAP_DIAMETERS,
    piece_diameter: float | None = None,
) -> list[float]:
    """Cluster detections into rows by travel coordinate (x).

    Returns one representative travel coordinate per row — the MEDIAN x of the
    row's members — sorted ascending. Empty input → empty list.

    ``detections`` are objects exposing ``.centroid`` (and ideally ``.width_px``
    / ``.height_px``). The split threshold is ``gap_diameters × piece_diameter``:
    an absolute distance, but derived from the detected dough SIZE, so slightly
    staggered/irregular pieces (within ~a piece diameter along travel) merge into
    one row, while a clearly larger gap starts the next. ``piece_diameter``
    defaults to the median detected diameter; pass it to override (e.g. a profile
    fallback when detections carry no size).
    """
    if not detections:
        return []
    xs = sorted(float(d.centroid[0]) for d in detections)
    if len(xs) == 1:
        return [xs[0]]

    if piece_diameter is None:
        piece_diameter = median_piece_diameter(detections)
    threshold = max(piece_diameter * gap_diameters, _GAP_EPS)

    rows: list[float] = []
    cluster: list[float] = [xs[0]]
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > threshold:
            rows.append(statistics.median(cluster))
            cluster = [xs[i]]
        else:
            cluster.append(xs[i])
    rows.append(statistics.median(cluster))
    return rows


@dataclass
class _TrackedRow:
    x: float          # current travel position of the row-line
    prev_x: float     # travel position on the previous frame it was seen
    fired: bool = False
    missed: int = 0


class RowLineTracker:
    """Stateful per-cycle tracker that fires once per row crossing.

    Reset at the start of every TuchabzugRunning cycle (see
    MainWindow._reset_tracking_session). ``update`` is fed the current frame's
    row-lines and returns how many DISTINCT rows crossed the transfer line this
    frame — each should produce one StopTuchabzug pulse.
    """

    def __init__(
        self,
        *,
        match_tol_factor: float = MATCH_TOL_FACTOR,
        max_missed: int = MAX_MISSED_FRAMES,
    ) -> None:
        self._rows: list[_TrackedRow] = []
        self._match_tol_factor = match_tol_factor
        self._max_missed = max_missed

    def reset(self) -> None:
        self._rows = []

    def _match_tol(self, row_xs: list[float]) -> float:
        """Distance within which a current row-line is the same tracked row.

        Relative to the median spacing between the current row-lines (no fixed
        pixels). With 0 or 1 row there is no spacing to measure, so matching is
        unrestricted (the single row always maps to the single tracked row).
        """
        if len(row_xs) < 2:
            return float("inf")
        s = sorted(row_xs)
        gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
        return max(statistics.median(gaps) * self._match_tol_factor, _GAP_EPS)

    def update(self, row_xs: list[float], transfer_x: float) -> int:
        """Advance the tracker by one frame; return rows that crossed this frame.

        A tracked row fires once, when the segment between its previous and
        current travel position straddles ``transfer_x`` (direction-agnostic).
        Already-fired rows never re-fire, so a new row behind a fired one — even
        with no gap — is tracked separately and fires on its own crossing.
        """
        tol = self._match_tol(row_xs)
        matched_idx: set[int] = set()

        # Match each current row-line to the nearest unmatched tracked row.
        for rx in row_xs:
            best_i = -1
            best_d = tol
            for i, tr in enumerate(self._rows):
                if i in matched_idx:
                    continue
                d = abs(tr.x - rx)
                if d <= best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                tr = self._rows[best_i]
                tr.prev_x = tr.x
                tr.x = rx
                tr.missed = 0
                matched_idx.add(best_i)
            else:
                # A brand-new row: prev_x == x so it cannot fire on first sight.
                self._rows.append(_TrackedRow(x=rx, prev_x=rx))
                matched_idx.add(len(self._rows) - 1)

        # Age out tracked rows that were not matched this frame.
        for i, tr in enumerate(self._rows):
            if i not in matched_idx:
                tr.missed += 1
                tr.prev_x = tr.x  # no movement observed
        self._rows = [tr for tr in self._rows if tr.missed <= self._max_missed]

        # Detect crossings on rows that actually moved this frame.
        fired = 0
        for tr in self._rows:
            if tr.fired or tr.prev_x == tr.x:
                continue
            if (tr.prev_x - transfer_x) * (tr.x - transfer_x) <= 0:
                tr.fired = True
                fired += 1
        return fired
