"""Tests for ``viscontrol.core.layout_quality`` (LayoutQualityMonitor).

No Qt. No camera. No PLC. No sleep(). Cluster inputs are built directly from
tangent_x lists (real ``ProximityCluster``/``ClusteredPiece`` objects — see
viscontrol/detection/proximity_clustering.py), or, for the pure debounce
tests (L4/L5/L8/L9/L10), directly-constructed ``LayoutQualityResult``
instances so debounce behavior is tested independently of clustering.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from viscontrol.core.layout_quality import LayoutQualityMonitor, LayoutQualityResult
from viscontrol.detection.proximity_clustering import (
    ClusteredPiece,
    Piece,
    ProximityCluster,
    cluster_by_tangent,
)

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class RecordingLogger:
    """Captures every structured log line the monitor emits, so tests can
    assert on prefixes without depending on the real loguru sink. Mirrors
    tests/test_transfer_orchestrator.py's RecordingLogger."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def _record(self, level: str, msg: str, *args) -> None:
        text = msg.format(*args) if args else msg
        self.lines.append((level, text))

    def debug(self, msg, *args) -> None: self._record("debug", msg, *args)  # noqa: E704
    def info(self, msg, *args) -> None: self._record("info", msg, *args)  # noqa: E704
    def warning(self, msg, *args) -> None: self._record("warning", msg, *args)  # noqa: E704
    def error(self, msg, *args) -> None: self._record("error", msg, *args)  # noqa: E704

    def has(self, needle: str) -> bool:
        return any(needle in text for _, text in self.lines)


class FakeStopSink:
    """Records every row_id handed to it, in order."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, row_id: int) -> None:
        self.calls.append(row_id)


def mk_cfg(**overrides) -> SimpleNamespace:
    defaults = dict(
        enabled=True,
        min_separation_ratio=1.8,
        ambiguous_frames_to_trigger=5,
        min_pieces_for_check=6,
        min_within_gap_px=5.0,
        error_code=2,
        auto_resume_on_good_layout=False,
        good_frames_to_resume=10,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def mk_clusters(rows: list[list[float]]) -> list[ProximityCluster]:
    """Build ProximityCluster objects directly from tangent_x lists — one
    list per row/cluster."""
    return [
        ProximityCluster(
            pieces=[
                ClusteredPiece(tangent_x=x, center_x=x, center_y=0.0, radius=0.0)
                for x in row
            ]
        )
        for row in rows
    ]


def mk_ambiguous_result(frame_id: int) -> LayoutQualityResult:
    return LayoutQualityResult(
        frame_id=frame_id, n_pieces=8, n_clusters=2,
        max_gap_within_cluster=120.0, min_gap_between_clusters=60.0,
        separation_ratio=0.5, is_ambiguous=True, reason="low_ratio",
        gaps=(60.0, 120.0),
    )


def mk_good_result(frame_id: int) -> LayoutQualityResult:
    return LayoutQualityResult(
        frame_id=frame_id, n_pieces=8, n_clusters=2,
        max_gap_within_cluster=80.0, min_gap_between_clusters=200.0,
        separation_ratio=2.5, is_ambiguous=False, reason="ok",
        gaps=(80.0, 200.0),
    )


# ---------------------------------------------------------------------------
# L1/L2/L3 — evaluate() against real logged measurements
# ---------------------------------------------------------------------------


def test_l1_real_normal_recorded():
    """REGRESSION: recorded normal load — must stay unambiguous."""
    mon = LayoutQualityMonitor(mk_cfg())
    clusters = mk_clusters([
        [102, 152, 152, 153, 201, 218, 232, 241],
        [477, 491, 495, 500, 580, 603, 604, 632],
        [834, 847, 851, 890, 892, 914, 923, 938],
    ])
    r = mon.evaluate(clusters, frame_id=1)
    assert r.max_gap_within_cluster == pytest.approx(80.0)
    assert r.min_gap_between_clusters == pytest.approx(202.0)
    assert r.separation_ratio == pytest.approx(2.525, abs=0.01)
    assert r.is_ambiguous is False
    assert r.reason == "ok"


def test_l2_real_normal_live():
    mon = LayoutQualityMonitor(mk_cfg())
    clusters = mk_clusters([[346, 351, 385], [602, 660, 671]])
    r = mon.evaluate(clusters, frame_id=1)
    assert r.max_gap_within_cluster == pytest.approx(58.0)
    assert r.min_gap_between_clusters == pytest.approx(217.0)
    assert r.separation_ratio == pytest.approx(3.74, abs=0.01)
    assert r.is_ambiguous is False
    assert r.reason == "ok"


def test_l3_real_staggered():
    """THE CASE THIS FEATURE EXISTS FOR. Piece values are exactly as given —
    not adjusted to make the test pass.

    cluster_by_tangent splits on strict '>' (see proximity_clustering.py),
    and the only gap near the described "tolerance 130" boundary is exactly
    130 (480-350) — that needs a tolerance strictly less than 130 to split,
    and every value in (95, 130) reproduces the identical partition
    ({280,350} | {480,565,660,780}) since the next-nearest gap is 95. 125 is
    used here as a representative value in that range.
    """
    mon = LayoutQualityMonitor(mk_cfg())
    pieces = [Piece(center_x=x, center_y=0.0, radius=0.0) for x in [280, 350, 480, 565, 660, 780]]
    clusters = cluster_by_tangent(pieces, tolerance_px=125)
    assert len(clusters) == 2  # sanity: this scenario needs an actual split
    r = mon.evaluate(clusters, frame_id=1)
    assert r.separation_ratio < 1.8
    assert r.is_ambiguous is True
    assert r.reason == "low_ratio"


# ---------------------------------------------------------------------------
# L4/L5 — debounce
# ---------------------------------------------------------------------------


def test_l4_debounce_requires_consecutive_frames():
    cfg = mk_cfg(ambiguous_frames_to_trigger=5)
    mon = LayoutQualityMonitor(cfg)

    for i in range(1, 5):  # frames 1..4: ambiguous, but not yet 5 in a row
        triggered = mon.update(mk_ambiguous_result(i))
        assert triggered is False
        assert mon.is_locked is False

    triggered = mon.update(mk_ambiguous_result(5))  # 5th consecutive
    assert triggered is True
    assert mon.is_locked is True

    triggered = mon.update(mk_ambiguous_result(6))  # already locked
    assert triggered is False
    assert mon.is_locked is True


def test_l5_single_bad_frame_does_not_lock():
    cfg = mk_cfg(ambiguous_frames_to_trigger=5)
    log = RecordingLogger()
    mon = LayoutQualityMonitor(cfg, log)

    for i in range(1, 51):
        is_amb = i % 2 == 0
        r = mk_ambiguous_result(i) if is_amb else mk_good_result(i)
        triggered = mon.update(r)
        assert triggered is False

    assert mon.is_locked is False
    assert log.has("LAYOUT-STREAK-RESET")


# ---------------------------------------------------------------------------
# L6/L7 — insufficient data / divide-by-zero guard
# ---------------------------------------------------------------------------


def test_l6_insufficient_data_never_locks():
    cfg = mk_cfg(ambiguous_frames_to_trigger=3, min_pieces_for_check=6)

    # (a) one cluster only, plenty of pieces — "nothing to separate".
    mon_a = LayoutQualityMonitor(cfg)
    clusters_a = mk_clusters([[100, 150, 200, 250, 300, 350, 400]])
    for i in range(1, 51):
        r = mon_a.evaluate(clusters_a, i)
        assert r.reason == "insufficient_data"
        assert r.is_ambiguous is False
        assert mon_a.update(r) is False
    assert mon_a.is_locked is False

    # (b) 3 pieces total across 2 clusters — below min_pieces_for_check.
    mon_b = LayoutQualityMonitor(cfg)
    clusters_b = mk_clusters([[100, 150], [400]])
    for i in range(1, 51):
        r = mon_b.evaluate(clusters_b, i)
        assert r.reason == "insufficient_data"
        assert r.is_ambiguous is False
        assert mon_b.update(r) is False
    assert mon_b.is_locked is False


def test_l7_identical_tangents_no_false_lock():
    cfg = mk_cfg(min_within_gap_px=5.0, min_pieces_for_check=6)
    mon = LayoutQualityMonitor(cfg)
    # One row of near-identical tangents (within_max < 5) plus a second,
    # far-away cluster so n_clusters >= 2 and n_pieces >= 6 (not
    # insufficient-data) — isolates the divide-by-near-zero guard itself.
    clusters = mk_clusters([[100.0, 100.5, 101.0, 101.5], [500.0, 500.3]])
    r = mon.evaluate(clusters, frame_id=1)
    assert r.max_gap_within_cluster < 5.0
    assert r.separation_ratio == float("inf")
    assert r.is_ambiguous is False
    assert r.reason == "ok"


# ---------------------------------------------------------------------------
# L8/L9 — clearing
# ---------------------------------------------------------------------------


def test_l8_lock_requires_explicit_clear():
    cfg = mk_cfg(ambiguous_frames_to_trigger=3, auto_resume_on_good_layout=False)
    log = RecordingLogger()
    mon = LayoutQualityMonitor(cfg, log)

    for i in range(1, 4):
        mon.update(mk_ambiguous_result(i))
    assert mon.is_locked is True

    for i in range(4, 104):  # 100 good frames
        triggered = mon.update(mk_good_result(i))
        assert triggered is False
    assert mon.is_locked is True  # auto_resume is False — still locked

    mon.clear_lock("operator_ack")
    assert mon.is_locked is False
    assert log.has("LAYOUT-LOCK-CLEARED")
    assert log.has("operator_ack")


def test_l9_auto_resume_when_enabled():
    cfg = mk_cfg(
        ambiguous_frames_to_trigger=3,
        auto_resume_on_good_layout=True,
        good_frames_to_resume=10,
    )
    log = RecordingLogger()
    mon = LayoutQualityMonitor(cfg, log)

    frame_id = 0
    for _ in range(3):
        frame_id += 1
        mon.update(mk_ambiguous_result(frame_id))
    assert mon.is_locked is True

    for _ in range(9):  # 9 good frames — not yet enough
        frame_id += 1
        mon.update(mk_good_result(frame_id))
        assert mon.is_locked is True

    frame_id += 1  # 10th consecutive good frame — auto-clears
    mon.update(mk_good_result(frame_id))
    assert mon.is_locked is False
    assert log.has("LAYOUT-LOCK-CLEARED")
    assert log.has("layout_recovered")


# ---------------------------------------------------------------------------
# L10 — blocked stop while locked (integration with a FakeStopSink)
# ---------------------------------------------------------------------------


def test_l10_blocked_stop_while_locked():
    """Mirrors the production guard pattern (MainWindow._orchestrator_stop_sink):
    'if monitor.is_locked: block + log; else: call the real sink.'"""
    cfg = mk_cfg(ambiguous_frames_to_trigger=2)
    log = RecordingLogger()
    mon = LayoutQualityMonitor(cfg, log)
    sink = FakeStopSink()

    def guarded_sink(row_id: int, frame_id: int) -> None:
        if mon.is_locked:
            log.warning(
                "[LAYOUT] LAYOUT-LOCK-BLOCKED-STOP frame={} row={} "
                "note=cloth_already_stopped_layout_fault_active",
                frame_id, row_id,
            )
            return
        sink(row_id)

    # Row 1 crosses the tripwire before any lock — sink called normally.
    guarded_sink(1, frame_id=1)
    assert sink.calls == [1]

    # Trigger the lock (2 consecutive ambiguous frames).
    mon.update(mk_ambiguous_result(2))
    mon.update(mk_ambiguous_result(3))
    assert mon.is_locked is True

    # Row 2 crosses the tripwire while locked — must be blocked, not sent.
    guarded_sink(2, frame_id=4)
    assert sink.calls == [1]
    assert log.has("LAYOUT-LOCK-BLOCKED-STOP")


# ---------------------------------------------------------------------------
# L11 — order independence
# ---------------------------------------------------------------------------


def test_l11_order_independence():
    cfg = mk_cfg()
    rows = [
        [102, 152, 152, 153, 201, 218, 232, 241],
        [477, 491, 495, 500, 580, 603, 604, 632],
        [834, 847, 851, 890, 892, 914, 923, 938],
    ]
    baseline = LayoutQualityMonitor(cfg).evaluate(mk_clusters(rows), frame_id=1)

    for seed in range(20):
        rnd = random.Random(seed)
        shuffled_rows = [list(row) for row in rows]
        for row in shuffled_rows:
            rnd.shuffle(row)
        rnd.shuffle(shuffled_rows)

        r = LayoutQualityMonitor(cfg).evaluate(mk_clusters(shuffled_rows), frame_id=1)
        assert r.separation_ratio == pytest.approx(baseline.separation_ratio)
        assert r.is_ambiguous == baseline.is_ambiguous
        assert r.max_gap_within_cluster == pytest.approx(baseline.max_gap_within_cluster)
        assert r.min_gap_between_clusters == pytest.approx(baseline.min_gap_between_clusters)
