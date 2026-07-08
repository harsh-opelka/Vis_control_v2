"""DIAGNOSTIC (logging-only): automatic column detection via Y-centroid learning.

Groundwork for a possible future column-based tracking mode. ``ColumnLearner``
watches the spread of detected pieces' Y-centroids over time and derives
``grid_columns`` Y-bands from it, purely for comparison against what Layer 1
(``group_by_gap``) and Layer 2 (sticky anchor tracking, see MainWindow.
_apply_tangent_stop_edge) already produce.

Entirely observational: nothing here is read by any fire/stop decision. See
MainWindow's COLUMN-LEARN logging (config.column_learning) for the call site.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from viscontrol.core.logger import logger


class ColumnLearner:
    """Rolling-window learner for column Y-bands.

    Feed every detected piece's Y-centroid each frame via :meth:`update`; it
    periodically (every ``recompute_every_n_frames`` calls to ``update``)
    re-derives ``grid_columns`` Y-bands by sorting the window and splitting at
    its ``grid_columns - 1`` largest gaps. ``bands`` holds the last computed
    result and persists across recomputations (never resets to empty once
    populated) so assignment always has something to work with between
    recomputes.
    """

    def __init__(
        self,
        grid_columns: int,
        *,
        window_size: int = 100,
        recompute_every_n_frames: int = 30,
    ) -> None:
        self._grid_columns = max(1, grid_columns)
        self._window: deque[float] = deque(maxlen=max(1, window_size))
        self._recompute_every_n_frames = max(1, recompute_every_n_frames)
        self._frames_since_recompute = 0
        self.bands: list[list[float]] = []
        self.total_pieces_observed = 0
        self.recompute_count = 0

    def update(self, y_centroids: Sequence[float]) -> None:
        """Feed one frame's worth of Y-centroids into the rolling window.

        Call once per frame with every detected piece's Y-centroid. Triggers
        a band recompute every ``recompute_every_n_frames`` calls, keeping
        bands stable rather than jittery instead of re-deriving them from
        scratch every single frame.
        """
        for y in y_centroids:
            self._window.append(float(y))
        self.total_pieces_observed += len(y_centroids)
        self._frames_since_recompute += 1
        if self._frames_since_recompute >= self._recompute_every_n_frames:
            self._frames_since_recompute = 0
            self.recompute_bands()

    def recompute_bands(self) -> list[list[float]]:
        """(Re)derive ``column_y_bands`` from the current window.

        Sorts the collected Y-centroids, finds the ``grid_columns - 1``
        largest gaps between consecutive values, and splits into
        ``grid_columns`` bands at those gaps. Leaves ``self.bands``
        unchanged (and returns it as-is) when fewer than ``grid_columns``
        pieces have been seen yet — there aren't enough gaps to split on.
        """
        ys = sorted(self._window)
        if len(ys) < self._grid_columns:
            logger.info(
                "ColumnLearner: insufficient data, {} pieces seen, need >= grid_columns",
                len(ys),
            )
            return self.bands

        gap_indices = sorted(
            range(len(ys) - 1), key=lambda i: ys[i + 1] - ys[i], reverse=True,
        )
        split_after = sorted(gap_indices[: self._grid_columns - 1])

        bands: list[list[float]] = []
        start = 0
        for idx in split_after:
            bands.append([ys[start], ys[idx]])
            start = idx + 1
        bands.append([ys[start], ys[-1]])

        self.bands = bands
        self.recompute_count += 1
        logger.info(
            "ColumnLearner: recomputed bands={} from {} samples (recompute #{})",
            bands, len(ys), self.recompute_count,
        )
        return bands

    def assign_column(self, y: float) -> tuple[int, bool]:
        """Column index for ``y``, and whether it fell inside a band.

        Returns ``(column_index, in_band)``. When ``y`` falls outside every
        band, ``column_index`` is the NEAREST band by center distance and
        ``in_band`` is False, so the caller can log a warning. Returns
        ``(-1, False)`` only when no bands have been computed yet.
        """
        if not self.bands:
            return -1, False
        for i, (y_min, y_max) in enumerate(self.bands):
            if y_min <= y <= y_max:
                return i, True
        centers = [(b[0] + b[1]) / 2.0 for b in self.bands]
        nearest = min(range(len(centers)), key=lambda i: abs(centers[i] - y))
        return nearest, False
