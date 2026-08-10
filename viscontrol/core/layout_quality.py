"""Layout-quality safety lock.

``cluster_by_tangent`` (see viscontrol/detection/proximity_clustering.py)
splits dough rows at gaps in ``tangent_x``. That only works when there is a
clear jump between "gap inside a row" and "gap between rows". When the load
is staggered enough that a within-row gap is *larger* than a between-row
gap, no ``tolerance_px`` value can separate them correctly and the grouping
flickers frame to frame. Transferring two merged rows into the fryer
together is worse than halting for the operator to re-space the dough — so
this module measures how separable the CURRENT clustering actually is and
raises a debounced lock when it is not.

This module is a pure measurement + debounce layer: it never re-clusters
and never changes ``cluster_by_tangent``, its tolerance, or any Hough
parameter. It consumes the SAME cluster list the production path already
computed (see MainWindow._run_transfer_orchestrator) and only looks at
``tangent_x`` — never ``center_y`` (Y is deliberately never part of this
feature, see the module's callers).

No Qt. No cv2. No PySide6. Fully unit-testable with fakes: no camera, no
PLC, no clock, no sleep().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class LayoutQualityConfig(Protocol):
    """Shape of the config object this module reads (see
    viscontrol.core.config._LayoutQualitySection). Duck-typed — any object
    with these attributes works, so tests don't need the real pydantic model.
    """

    enabled: bool
    min_separation_ratio: float
    ambiguous_frames_to_trigger: int
    min_pieces_for_check: int
    min_within_gap_px: float
    error_code: int
    auto_resume_on_good_layout: bool
    good_frames_to_resume: int


class _LoggerLike(Protocol):
    def debug(self, *args: Any, **kwargs: Any) -> None: ...
    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, *args: Any, **kwargs: Any) -> None: ...
    def error(self, *args: Any, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class LayoutQualityResult:
    """One frame's layout-separability measurement.

    ``gaps`` is every neighbour gap (within AND between clusters) sorted
    ascending — kept for logging (LAYOUT-LOCK-TRIGGERED), not for further
    computation.
    """

    frame_id: int
    n_pieces: int
    n_clusters: int
    max_gap_within_cluster: float
    min_gap_between_clusters: float
    separation_ratio: float
    is_ambiguous: bool
    reason: str  # "ok" | "low_ratio" | "insufficient_data"
    gaps: tuple[float, ...]


class LayoutQualityMonitor:
    """Measures per-frame layout separability and debounces it into a
    latched lock.

    ``evaluate()`` is a pure measurement of one frame's cluster list — see
    its docstring for the exact rule and the real-log verification numbers.

    ``update()`` feeds a measurement into a consecutive-ambiguous-frame
    counter and is the ONLY place the lock is triggered. Single-frame noise
    must never trigger it; that's what ``cfg.ambiguous_frames_to_trigger``
    debounces (this is what stops the old row-splitting flicker from simply
    reappearing as a flickering STOP/lock).

    Once triggered, the lock is latched (``is_locked`` stays True) until
    ``clear_lock()`` is called explicitly — a single good frame must never
    silently clear it when ``cfg.auto_resume_on_good_layout`` is False. When
    that flag IS True, ``update()`` itself calls ``clear_lock(
    "layout_recovered")`` once ``cfg.good_frames_to_resume`` consecutive
    fresh updates come back non-ambiguous.
    """

    def __init__(self, cfg: LayoutQualityConfig, logger: _LoggerLike | None = None) -> None:
        self._cfg = cfg
        self._logger = logger

        self._consecutive_ambiguous_frames: int = 0
        self._consecutive_good_frames: int = 0
        self._is_locked: bool = False
        self._locked_since_frame: int | None = None
        self._last_frame_id: int = 0

        # Session-scoped stats, for LAYOUT-SESSION-SUMMARY on shutdown.
        self._locks_triggered: int = 0
        self._total_locked_frames: int = 0
        self._min_ratio_seen: float = float("inf")
        self._max_ratio_seen: float = float("-inf")
        self._frames_evaluated: int = 0

    # ---------- properties ----------

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @property
    def consecutive_ambiguous_frames(self) -> int:
        """Current ambiguous-frame streak — for the LAYOUT-QUALITY log line."""
        return self._consecutive_ambiguous_frames

    # ---------- lifecycle ----------

    def reset(self) -> None:
        """Call wherever the orchestrator's start_cycle() is legitimately
        called (new TuchabzugRunning pull, STOP button, wizard completion) —
        never on a raw PLC edge.

        Resets the debounce counters for the new cycle only. Deliberately
        does NOT touch ``is_locked``: a fresh cycle must never silently
        clear an active lock — only ``clear_lock()`` may do that (see class
        docstring).
        """
        self._consecutive_ambiguous_frames = 0
        self._consecutive_good_frames = 0

    def clear_lock(self, reason: str) -> None:
        """Explicit unlock. ``reason`` is a free-form string for the log
        (typically "operator_ack" or "layout_recovered")."""
        locked_for = (
            self._last_frame_id - self._locked_since_frame
            if self._locked_since_frame is not None else 0
        )
        self._is_locked = False
        self._locked_since_frame = None
        self._consecutive_ambiguous_frames = 0
        self._consecutive_good_frames = 0
        if self._logger is not None:
            self._logger.warning(
                "[LAYOUT] LAYOUT-LOCK-CLEARED frame={} reason={} locked_for_frames={}",
                self._last_frame_id, reason, locked_for,
            )

    # ---------- measurement ----------

    def evaluate(self, clusters: Sequence[Any], frame_id: int) -> LayoutQualityResult:
        """Pure measurement of THIS frame's cluster list. Never re-clusters,
        never mutates ``self``.

        Rule:
          1. Collect every piece's tangent_x from every cluster; sort ascending.
          2. Compute neighbour gaps between consecutive sorted values.
          3. Classify each gap WITHIN (both pieces in the same cluster) or
             BETWEEN (different clusters).
          4. max_gap_within_cluster = max(WITHIN gaps), 0.0 if none.
             min_gap_between_clusters = min(BETWEEN gaps), +inf if none.
          5. separation_ratio = between / within, guarded: if
             max_gap_within_cluster < cfg.min_within_gap_px, ratio = +inf
             (a row with near-identical tangents is unambiguous by
             definition — dividing by a tiny number would otherwise produce
             a meaningless huge ratio, or ZeroDivisionError at exactly 0).
          6. INSUFFICIENT DATA (never ambiguous — thin evidence, e.g. a
             partly-visible cloth at the start of a batch, is normal, not a
             fault): n_clusters < 2, or n_pieces < cfg.min_pieces_for_check.
          7. Otherwise is_ambiguous = separation_ratio < cfg.min_separation_ratio.

        Verified against real logged measurements:
            within=80,  between=202 -> ratio 2.52 -> OK        (recorded normal)
            within=58,  between=217 -> ratio 3.74 -> OK        (live normal)
            within=120, between=95  -> ratio 0.79 -> AMBIGUOUS (hand-staggered)
        """
        cfg = self._cfg

        entries: list[tuple[float, int]] = []
        for cluster_index, cluster in enumerate(clusters):
            for piece in cluster.pieces:
                entries.append((float(piece.tangent_x), cluster_index))
        entries.sort(key=lambda e: e[0])

        n_pieces = len(entries)
        n_clusters = len(clusters)

        within_gaps: list[float] = []
        between_gaps: list[float] = []
        all_gaps: list[float] = []
        for (x0, c0), (x1, c1) in zip(entries, entries[1:]):
            gap = x1 - x0
            all_gaps.append(gap)
            if c0 == c1:
                within_gaps.append(gap)
            else:
                between_gaps.append(gap)

        max_within = max(within_gaps) if within_gaps else 0.0
        min_between = min(between_gaps) if between_gaps else float("inf")

        # Guard against dividing by ~0: a tiny epsilon floor on top of the
        # configured min_within_gap_px keeps this safe even if that config
        # value is itself 0.
        if max_within < max(cfg.min_within_gap_px, 1e-9):
            separation_ratio = float("inf")
        else:
            separation_ratio = min_between / max_within

        insufficient = n_clusters < 2 or n_pieces < cfg.min_pieces_for_check
        if insufficient:
            is_ambiguous = False
            reason = "insufficient_data"
        else:
            is_ambiguous = separation_ratio < cfg.min_separation_ratio
            reason = "low_ratio" if is_ambiguous else "ok"

        return LayoutQualityResult(
            frame_id=frame_id,
            n_pieces=n_pieces,
            n_clusters=n_clusters,
            max_gap_within_cluster=max_within,
            min_gap_between_clusters=min_between,
            separation_ratio=separation_ratio,
            is_ambiguous=is_ambiguous,
            reason=reason,
            gaps=tuple(all_gaps),
        )

    # ---------- debounce ----------

    def update(self, result: LayoutQualityResult) -> bool:
        """Feeds one measurement into the consecutive-ambiguous-frame
        counter. Returns True ONLY on the frame the lock should TRIGGER
        (i.e. at most once per lock — re-evaluating while already locked
        never returns True again until clear_lock() is called).

        Only ever called once per fresh Hough/observation update — never on
        a cached/stale frame (the caller enforces this, same convention as
        PieceConfirmationTracker.update()).
        """
        self._frames_evaluated += 1
        self._last_frame_id = result.frame_id

        ratio = result.separation_ratio
        if ratio != float("inf"):
            if ratio < self._min_ratio_seen:
                self._min_ratio_seen = ratio
            if ratio > self._max_ratio_seen:
                self._max_ratio_seen = ratio

        if result.is_ambiguous:
            self._consecutive_ambiguous_frames += 1
            self._consecutive_good_frames = 0
        else:
            if self._consecutive_ambiguous_frames > 0 and self._logger is not None:
                self._logger.debug(
                    "[LAYOUT] LAYOUT-STREAK-RESET frame={} was={} ratio={:.2f}",
                    result.frame_id, self._consecutive_ambiguous_frames, ratio,
                )
            self._consecutive_ambiguous_frames = 0
            self._consecutive_good_frames += 1

        triggered = False
        if (
            not self._is_locked
            and self._consecutive_ambiguous_frames >= self._cfg.ambiguous_frames_to_trigger
        ):
            self._is_locked = True
            self._locked_since_frame = result.frame_id
            self._locks_triggered += 1
            triggered = True

        if self._is_locked:
            self._total_locked_frames += 1
            if (
                not triggered
                and self._cfg.auto_resume_on_good_layout
                and self._consecutive_good_frames >= self._cfg.good_frames_to_resume
            ):
                self.clear_lock("layout_recovered")

        return triggered

    # ---------- shutdown ----------

    def log_session_summary(self) -> None:
        """LAYOUT-SESSION-SUMMARY at INFO — call once on shutdown."""
        if self._logger is None:
            return
        min_ratio = self._min_ratio_seen if self._min_ratio_seen != float("inf") else 0.0
        max_ratio = self._max_ratio_seen if self._max_ratio_seen != float("-inf") else 0.0
        self._logger.info(
            "[LAYOUT] LAYOUT-SESSION-SUMMARY locks_triggered={} total_locked_frames={} "
            "min_ratio_seen={:.2f} max_ratio_seen={:.2f} frames_evaluated={}",
            self._locks_triggered, self._total_locked_frames,
            min_ratio, max_ratio, self._frames_evaluated,
        )
