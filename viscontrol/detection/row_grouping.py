"""Row/grid grouping helpers — retired from the fire decision.

StopTuchabzug firing has moved to a grid-free tangent-based proximity
cluster model (see viscontrol/detection/proximity_clustering.py:
cluster_by_tangent, and viscontrol/core/transfer_orchestrator.py).
``group_by_gap``, ``group_rows``, ``group_into_rows``,
``reject_group_outliers``, and ``RowLineTracker`` — no longer wired into
any fire decision — have been removed from this module.

``leading_edge_x`` and ``median_piece_diameter`` are still live: they're
used by both the legacy tripwire RowPhase tracking
(MainWindow._track_row_phase) and the cluster-based fire decision.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Sequence

# Row split tolerance, as a multiple of the median DETECTED piece diameter:
# pieces whose travel coordinates differ by less than
# (gap_diameters × median diameter) are the same row; a clearly larger gap
# starts the next row. Tolerance scales with dough size (no fixed pixels) and
# is overridable per call via the gap_diameters parameter below.
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


# Grouping outlier rejection: within a sliced group, a member whose leading-edge
# X differs from the median of the OTHER members by more than this many pixels
# is treated as a straggler (e.g. a leftover piece that slipped past the
# boundary filter) rather than a genuine member of that row.
GROUP_OUTLIER_MAX_DIST_PX: float = 100.0


def row_leading_edge(row: Sequence[Any]) -> float:
    """Representative leading-edge travel position of a row (median of members,
    robust to one outlier piece)."""
    if not row:
        return 0.0
    return statistics.median(leading_edge_x(d) for d in row)


@dataclass
class _TrackedRow:
    x: float          # current travel position of the row-line
    prev_x: float     # travel position on the previous frame it was seen
    fired: bool = False
    missed: int = 0
