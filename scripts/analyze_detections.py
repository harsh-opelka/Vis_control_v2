"""Offline analysis of a detection_recorder.py JSONL recording.

Read-only: loads a recording via viscontrol.core.detection_recorder and
prints a plain-text diagnostic report to stdout. Makes no changes to
viscontrol/, writes no files, does no plotting.

The per-update pairing used here (section 2) is a simple greedy
nearest-2D-distance matcher for MEASUREMENT purposes only — it is NOT the
production matcher (see viscontrol/core/piece_tracker.py: PieceTracker,
which matches on 2D distance + a Y penalty + a radius penalty, with proper
one-to-one assignment and configured gates). This script exists to inform
what those gates (piece_match_max_dist_px, piece_match_max_dy_px,
roi_valid_y_min/max, etc.) should be set to, not to replace them.

Usage:
    py scripts/analyze_detections.py logs/detections_20260803_143000.jsonl
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viscontrol.core.detection_recorder import load_recording  # noqa: E402

_BUCKET_WIDTH_PX = 25.0
_BAR_WIDTH = 50


# ---------------------------------------------------------------------------
# small stats helpers (stdlib only)
# ---------------------------------------------------------------------------


def _percentile(values: list, p: float) -> float:
    """Linear-interpolation percentile (numpy-style), 0 <= p <= 100."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _print_bar_histogram(counts: dict, *, value_fmt=str) -> None:
    if not counts:
        print("    (no data)")
        return
    max_count = max(counts.values())
    for key in sorted(counts.keys()):
        n = counts[key]
        bar_len = int(round((n / max_count) * _BAR_WIDTH)) if max_count else 0
        print(f"    {value_fmt(key):>14}: {'#' * bar_len} ({n})")


# ---------------------------------------------------------------------------
# greedy 2D nearest-distance pairing (measurement only — see module docstring)
# ---------------------------------------------------------------------------


def _greedy_pair(dets_a: list, dets_b: list) -> list:
    candidates = []
    for i, a in enumerate(dets_a):
        for j, b in enumerate(dets_b):
            dx = b["center_x"] - a["center_x"]
            dy = b["center_y"] - a["center_y"]
            dist = math.hypot(dx, dy)
            candidates.append((dist, i, j))
    candidates.sort(key=lambda c: c[0])
    used_a: set = set()
    used_b: set = set()
    pairs = []
    for _dist, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((dets_a[i], dets_b[j]))
    return pairs


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def section_frame_cadence(frames: list) -> None:
    print("=" * 78)
    print("1. FRAME CADENCE")
    print("=" * 78)
    print(f"  fresh detection frames: {len(frames)}")

    if len(frames) >= 2:
        intervals_ms = [
            (b.ts - a.ts) * 1000.0 for a, b in zip(frames, frames[1:])
        ]
        print(
            f"  wall-clock interval between fresh frames (ms): "
            f"median={_fmt(statistics.median(intervals_ms))} "
            f"p95={_fmt(_percentile(intervals_ms, 95))} "
            f"min={_fmt(min(intervals_ms))} max={_fmt(max(intervals_ms))}"
        )
    else:
        print("  wall-clock interval: n/a (need >= 2 frames)")

    counts = [len(f.detections) for f in frames]
    if counts:
        print(
            f"  detections per frame: min={min(counts)} "
            f"median={statistics.median(counts):.1f} max={max(counts)}"
        )
    print("  histogram (detections per frame -> frame count):")
    hist: dict = {}
    for c in counts:
        hist[c] = hist.get(c, 0) + 1
    _print_bar_histogram(hist)
    print()


def section_per_update_displacement(frames: list) -> dict:
    print("=" * 78)
    print("2. PER-UPDATE DISPLACEMENT")
    print("=" * 78)

    dxs: list = []
    dys: list = []
    euclids: list = []
    drs: list = []
    all_pairs: list = []

    for a, b in zip(frames, frames[1:]):
        pairs = _greedy_pair(a.detections, b.detections)
        all_pairs.extend(pairs)
        for da, db in pairs:
            dx = db["center_x"] - da["center_x"]
            dy = db["center_y"] - da["center_y"]
            dxs.append(dx)
            dys.append(dy)
            euclids.append(math.hypot(dx, dy))
            drs.append(db["radius"] - da["radius"])

    if not dxs:
        print("  no consecutive-frame pairs available (need >= 2 frames, each with detections)")
        print()
        return {"pairs": [], "dxs": [], "dys": [], "euclids": [], "drs": []}

    print(f"  matched pairs across all consecutive fresh-frame transitions: {len(dxs)}")
    print(
        f"  dx (px):      median={_fmt(statistics.median(dxs))} "
        f"p5={_fmt(_percentile(dxs, 5))} p95={_fmt(_percentile(dxs, 95))} "
        f"min={_fmt(min(dxs))} max={_fmt(max(dxs))}"
    )
    print(
        f"  dy (px):      median={_fmt(statistics.median(dys))} "
        f"p5={_fmt(_percentile(dys, 5))} p95={_fmt(_percentile(dys, 95))} "
        f"min={_fmt(min(dys))} max={_fmt(max(dys))}"
    )
    print(
        f"  euclid (px):  median={_fmt(statistics.median(euclids))} "
        f"p90={_fmt(_percentile(euclids, 90))} p95={_fmt(_percentile(euclids, 95))} "
        f"p99={_fmt(_percentile(euclids, 99))} max={_fmt(max(euclids))}"
    )

    p99_euclid = _percentile(euclids, 99)
    abs_dys = [abs(v) for v in dys]
    p99_abs_dy = _percentile(abs_dys, 99)
    print()
    print(f'  RECOMMENDATION: piece_match_max_dist_px should be >= p99 euclid = {_fmt(p99_euclid)}')
    print(f'  RECOMMENDATION: piece_match_max_dy_px should be >= p99 |dy| = {_fmt(p99_abs_dy)}')
    print()

    return {"pairs": all_pairs, "dxs": dxs, "dys": dys, "euclids": euclids, "drs": drs}


def section_radius_stability(frames: list, displacement: dict) -> None:
    print("=" * 78)
    print("3. RADIUS STABILITY")
    print("=" * 78)

    all_radii = [d["radius"] for f in frames for d in f.detections]
    if all_radii:
        print(
            f"  radius across whole run (px): min={_fmt(min(all_radii))} "
            f"median={_fmt(statistics.median(all_radii))} max={_fmt(max(all_radii))} "
            f"stddev={_fmt(statistics.pstdev(all_radii))}"
        )
    else:
        print("  radius across whole run: n/a (no detections)")

    drs = displacement.get("drs", [])
    if drs:
        abs_drs = [abs(v) for v in drs]
        print(
            f"  |dr| per update (px):    median={_fmt(statistics.median(abs_drs))} "
            f"p95={_fmt(_percentile(abs_drs, 95))}"
        )
    else:
        print("  |dr| per update: n/a (no paired detections)")

    pairs = displacement.get("pairs", [])
    if pairs:
        cx_values = []
        tangent_values = []
        for da, db in pairs:
            for d in (da, db):
                cx_values.append(d["center_x"])
                tangent_values.append(d["center_x"] - d["radius"])
        var_cx = statistics.pvariance(cx_values)
        var_tangent = statistics.pvariance(tangent_values)
        print(
            f"  variance over paired detections: center_x={_fmt(var_cx)} "
            f"(center_x - radius)={_fmt(var_tangent)}  "
            f"[ratio {_fmt(var_tangent / var_cx) if var_cx else float('nan')}x]"
        )

        dxs = displacement.get("dxs", [])
        if dxs and drs:
            tangent_deltas = [dx - dr for dx, dr in zip(dxs, drs)]
            var_dx = statistics.pvariance(dxs)
            var_tangent_delta = statistics.pvariance(tangent_deltas)
            ratio = (var_tangent_delta / var_dx) if var_dx else float("nan")
            print(
                f"  (delta-based, isolates jitter from real motion) "
                f"variance of per-update dx={_fmt(var_dx)} "
                f"vs per-update d(tangent_x)={_fmt(var_tangent_delta)}  [ratio {_fmt(ratio)}x]"
            )
    else:
        print("  variance comparison: n/a (no paired detections)")
    print()


def section_y_distribution(frames: list) -> None:
    print("=" * 78)
    print("4. Y DISTRIBUTION")
    print("=" * 78)

    all_dets = [(f.frame_id, d) for f in frames for d in f.detections]
    if not all_dets:
        print("  no detections in this recording")
        print()
        return

    bucket_of = lambda cy: math.floor(cy / _BUCKET_WIDTH_PX)  # noqa: E731
    hist: dict = {}
    for _fid, d in all_dets:
        b = bucket_of(d["center_y"])
        hist[b] = hist.get(b, 0) + 1

    print(f"  histogram (center_y bucket, {_BUCKET_WIDTH_PX:.0f}px wide -> count):")
    _print_bar_histogram(
        hist,
        value_fmt=lambda b: f"[{b * _BUCKET_WIDTH_PX:.0f},{(b + 1) * _BUCKET_WIDTH_PX:.0f})",
    )

    total = len(all_dets)
    target = math.ceil(0.95 * total)
    min_bucket = min(hist.keys())
    max_bucket = max(hist.keys())
    n_buckets = max_bucket - min_bucket + 1
    counts_arr = [hist.get(min_bucket + i, 0) for i in range(n_buckets)]

    best_width = None
    best_range = (min_bucket, max_bucket)
    left = 0
    running = 0
    for right in range(n_buckets):
        running += counts_arr[right]
        while running >= target and left <= right:
            width = right - left + 1
            if best_width is None or width < best_width:
                best_width = width
                best_range = (min_bucket + left, min_bucket + right)
            running -= counts_arr[left]
            left += 1

    band_min = best_range[0] * _BUCKET_WIDTH_PX
    band_max = (best_range[1] + 1) * _BUCKET_WIDTH_PX

    print()
    print(
        f"  densest contiguous band containing >= 95% of detections: "
        f"[{band_min:.0f}, {band_max:.0f}) px"
    )

    outliers = [
        (fid, d) for fid, d in all_dets
        if d["center_y"] < band_min or d["center_y"] >= band_max
    ]
    print(f"  detections outside that band: {len(outliers)}")
    for fid, d in outliers:
        print(
            f"    frame={fid} center_x={d['center_x']:.1f} "
            f"center_y={d['center_y']:.1f} radius={d['radius']:.1f}"
        )

    print()
    print(f"  RECOMMENDATION: roi_valid_y_min = {band_min:.0f}")
    print(f"  RECOMMENDATION: roi_valid_y_max = {band_max:.0f}")
    print()


def section_travel_runway(frames: list, displacement: dict) -> None:
    print("=" * 78)
    print("5. TRAVEL RUNWAY")
    print("=" * 78)

    all_tangents = [
        d["center_x"] - d["radius"] for f in frames for d in f.detections
    ]
    if not all_tangents:
        print("  no detections in this recording")
        print()
        return

    max_tangent = max(all_tangents)
    min_tangent = min(all_tangents)
    print(f"  tangent_x observed: min={_fmt(min_tangent)} max={_fmt(max_tangent)}")

    transfer_xs = {f.transfer_x for f in frames}
    if len(transfer_xs) > 1:
        print(
            f"  WARNING: transfer_x varies across this recording ({sorted(transfer_xs)}); "
            f"using the last frame's value below."
        )
    transfer_x = frames[-1].transfer_x
    print(f"  transfer_x (from recording): {_fmt(transfer_x)}")

    dxs = displacement.get("dxs", [])
    if not dxs:
        print("  runway: n/a (no per-update dx available — see section 2)")
        print()
        return

    median_dx = statistics.median(dxs)
    print(f"  median per-update dx (from section 2): {_fmt(median_dx)}")

    if abs(median_dx) < 1e-9:
        print("  a row gets approximately UNDEFINED fresh updates of runway (median dx ~= 0)")
    else:
        runway = (max_tangent - transfer_x) / abs(median_dx)
        n_updates = max(0, round(runway))
        print(f"  a row gets approximately {n_updates} fresh updates of runway")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only analysis of a viscontrol detection_recorder.py JSONL recording.",
    )
    parser.add_argument("path", help="path to logs/detections_*.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1

    frames = load_recording(path)
    frames = [f for f in frames if f.is_fresh]
    frames.sort(key=lambda f: f.frame_id)

    print(f"loaded {len(frames)} fresh frames from {path}")
    print()

    if not frames:
        print("nothing to analyze — empty or all-stale recording")
        return 0

    section_frame_cadence(frames)
    displacement = section_per_update_displacement(frames)
    section_radius_stability(frames, displacement)
    section_y_distribution(frames)
    section_travel_runway(frames, displacement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
