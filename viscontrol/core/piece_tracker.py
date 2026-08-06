"""Stable per-piece identity tracking beneath the tangent-clustering layer.

Matches this frame's fresh detections against live tracks in 2D
(center_x/center_y/radius — NEVER tangent_x, which inherits the full Hough
radius-estimation error and is exactly the noise this module exists to
filter out before anything downstream sees it), then EMA-smooths each
matched track's geometry. ``tangent_x`` on a :class:`PieceTrack` is always
derived from the SMOOTHED center_x/radius, which is what makes it stable
enough for row grouping (see viscontrol/core/row_tracker.py) to use.

Pure Python + stdlib only (no numpy/scipy dependency — scipy is not a
declared project dependency, so the one-to-one assignment below is a
deterministic greedy-by-lowest-cost solver instead of
``linear_sum_assignment``; see PieceTracker._assign). No Qt. No cv2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawDetection:
    center_x: float
    center_y: float
    radius: float
    source_frame_id: int
    det_index: int  # per-frame array index ONLY. Never identity.


@dataclass
class PieceTrack:
    piece_id: int  # PERMANENT. Never reused within a process run.
    cycle_id: int
    # raw last observation
    raw_center_x: float
    raw_center_y: float
    raw_radius: float
    # smoothed state — this is what everything downstream uses
    center_x: float
    center_y: float
    radius: float
    tangent_x: float  # = center_x - radius, from SMOOTHED values
    prev_tangent_x: float
    # bookkeeping
    first_seen_frame: int
    last_seen_frame: int
    fresh_observations: int  # count of FRESH detection updates only
    missed_frames: int
    row_id: int | None = None  # assigned by RowTracker; at most one row
    velocity_x: float = 0.0  # px per fresh update, EMA. Diagnostic only.


@dataclass
class PieceUpdateResult:
    tracks: list  # list[PieceTrack] — all live tracks, smoothed
    created: list  # piece_ids created this update
    matched: list  # (piece_id, det_index, cost)
    missed: list  # piece_ids with no detection this update
    dropped: list  # piece_ids removed this update
    ambiguous: list  # (det_index, [candidate piece_ids], [costs], chosen)


class PieceTracker:
    """Tracks individual dough pieces by stable identity across frames.

    ``update()`` may ONLY be called with FRESH Hough results for this
    frame — see the assertion at the call site in
    MainWindow._run_transfer_orchestrator. A cached/redisplayed frame must
    never reach this, because ``fresh_observations`` is the counter that
    gates row confirmation in RowTracker.
    """

    def __init__(self, cfg, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger
        self._tracks: dict[int, PieceTrack] = {}
        self._next_piece_id: int = 1
        self._cycle_id: int = 0

    def reset(self, cycle_id: int) -> None:
        self._tracks = {}
        self._cycle_id = cycle_id
        # _next_piece_id is NEVER reset — piece_id stays permanent process-wide.

    # ---------- public API ----------

    def update(self, detections: list[RawDetection], frame_id: int) -> PieceUpdateResult:
        cfg = self._cfg
        live_tracks = list(self._tracks.values())

        eligible, det_candidates = self._build_cost_matrix(live_tracks, detections)
        chosen_for_det = self._assign(live_tracks, detections, eligible)

        matched: list[tuple[int, int, float]] = []
        ambiguous: list[tuple[int, list[int], list[float], int]] = []
        assigned_track_idx: set[int] = set()

        for di in sorted(chosen_for_det.keys(), key=lambda d: detections[d].det_index):
            ti, cost = chosen_for_det[di]
            assigned_track_idx.add(ti)
            track = live_tracks[ti]
            det = detections[di]
            matched.append((track.piece_id, det.det_index, cost))

            candidates = det_candidates.get(di, [])
            within_margin = [(c, t) for c, t in candidates if c <= cost + cfg.piece_ambiguous_margin]
            if len(within_margin) > 1:
                cand_sorted = sorted(
                    within_margin, key=lambda ct: (ct[0], live_tracks[ct[1]].piece_id)
                )
                candidate_ids = [live_tracks[t].piece_id for _, t in cand_sorted]
                costs = [round(c, 2) for c, _ in cand_sorted]
                ambiguous.append((det.det_index, candidate_ids, costs, track.piece_id))
                self._log(
                    "PIECE-AMBIGUOUS-MATCH", level="warning",
                    frame=frame_id, det_idx=det.det_index, candidates=candidate_ids,
                    costs=costs, chosen=track.piece_id, margin=cfg.piece_ambiguous_margin,
                )

            self._apply_match(track, det, frame_id, cost)

        missed: list[int] = []
        for ti, track in enumerate(live_tracks):
            if ti in assigned_track_idx:
                continue
            track.missed_frames += 1
            missed.append(track.piece_id)
            self._log(
                "PIECE-MISS", level="debug", frame=frame_id, piece=track.piece_id,
                missed=track.missed_frames, last_cx=track.center_x, last_cy=track.center_y,
                row=track.row_id,
            )

        # New tracks: sort unmatched detections deterministically (by
        # center_y/center_x, NOT input list position) so piece_id
        # assignment is identical regardless of detection-array order.
        unmatched_det_idx = [di for di in range(len(detections)) if di not in chosen_for_det]
        unmatched_det_idx.sort(key=lambda di: (detections[di].center_y, detections[di].center_x))

        created: list[int] = []
        for di in unmatched_det_idx:
            track = self._create_track(detections[di], frame_id)
            created.append(track.piece_id)

        dropped: list[int] = []
        for piece_id, track in list(self._tracks.items()):
            if track.missed_frames > cfg.piece_max_missed_frames:
                dropped.append(piece_id)
                self._log(
                    "PIECE-DROP", level="info", frame=frame_id, piece=piece_id,
                    missed=track.missed_frames, reason="max_missed_frames_exceeded",
                    row=track.row_id,
                )
                del self._tracks[piece_id]

        tracks_sorted = sorted(self._tracks.values(), key=lambda t: t.piece_id)
        return PieceUpdateResult(
            tracks=tracks_sorted, created=created, matched=matched,
            missed=missed, dropped=dropped, ambiguous=ambiguous,
        )

    # ---------- matching internals ----------

    def _build_cost_matrix(
        self, live_tracks: list[PieceTrack], detections: list[RawDetection],
    ) -> tuple[list[tuple[float, int, int]], dict[int, list[tuple[float, int]]]]:
        cfg = self._cfg
        eligible: list[tuple[float, int, int]] = []
        for ti, track in enumerate(live_tracks):
            for di, det in enumerate(detections):
                dx = det.center_x - track.center_x
                dy = det.center_y - track.center_y
                dr = abs(det.radius - track.radius)
                euclid = math.hypot(dx, dy)
                if euclid > cfg.piece_match_max_dist_px or abs(dy) > cfg.piece_match_max_dy_px:
                    continue
                cost = (
                    cfg.piece_cost_w_xy * euclid
                    + cfg.piece_cost_w_y * abs(dy)
                    + cfg.piece_cost_w_r * dr
                )
                eligible.append((cost, ti, di))

        det_candidates: dict[int, list[tuple[float, int]]] = {}
        for cost, ti, di in eligible:
            det_candidates.setdefault(di, []).append((cost, ti))
        return eligible, det_candidates

    def _assign(
        self,
        live_tracks: list[PieceTrack],
        detections: list[RawDetection],
        eligible: list[tuple[float, int, int]],
    ) -> dict[int, tuple[int, float]]:
        """Deterministic greedy-by-lowest-cost one-to-one assignment.

        Ties are broken by (track.piece_id, det.center_y, det.center_x) —
        never by list position — so the result is identical regardless of
        the order detections are supplied in (required by Test P3/F).
        """
        ordered = sorted(
            eligible,
            key=lambda item: (
                item[0],
                live_tracks[item[1]].piece_id,
                detections[item[2]].center_y,
                detections[item[2]].center_x,
            ),
        )
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        chosen_for_det: dict[int, tuple[int, float]] = {}
        for cost, ti, di in ordered:
            if ti in assigned_tracks or di in assigned_dets:
                continue
            assigned_tracks.add(ti)
            assigned_dets.add(di)
            chosen_for_det[di] = (ti, cost)
        return chosen_for_det

    # ---------- track lifecycle ----------

    def _create_track(self, det: RawDetection, frame_id: int) -> PieceTrack:
        piece_id = self._next_piece_id
        self._next_piece_id += 1
        tangent_x = det.center_x - det.radius
        track = PieceTrack(
            piece_id=piece_id,
            cycle_id=self._cycle_id,
            raw_center_x=det.center_x, raw_center_y=det.center_y, raw_radius=det.radius,
            center_x=det.center_x, center_y=det.center_y, radius=det.radius,
            tangent_x=tangent_x, prev_tangent_x=tangent_x,
            first_seen_frame=frame_id, last_seen_frame=frame_id,
            fresh_observations=1, missed_frames=0,
        )
        self._tracks[piece_id] = track
        self._log(
            "PIECE-CREATE", level="info", frame=frame_id, piece=piece_id,
            cx=track.center_x, cy=track.center_y, r=track.radius, tan=track.tangent_x,
            det_idx=det.det_index,
        )
        return track

    def _apply_match(self, track: PieceTrack, det: RawDetection, frame_id: int, cost: float) -> None:
        cfg = self._cfg
        a = cfg.piece_ema_alpha_pos
        ar = cfg.piece_ema_alpha_radius

        old_cx, old_cy, old_r, old_tan = track.center_x, track.center_y, track.radius, track.tangent_x

        track.raw_center_x = det.center_x
        track.raw_center_y = det.center_y
        track.raw_radius = det.radius
        track.center_x = a * det.center_x + (1 - a) * track.center_x
        track.center_y = a * det.center_y + (1 - a) * track.center_y
        track.radius = ar * det.radius + (1 - ar) * track.radius
        track.prev_tangent_x = track.tangent_x
        track.tangent_x = track.center_x - track.radius
        track.velocity_x = a * (track.center_x - old_cx) + (1 - a) * track.velocity_x
        track.fresh_observations += 1
        track.missed_frames = 0
        track.last_seen_frame = frame_id

        self._log(
            "PIECE-MATCH", level="debug", frame=frame_id, piece=track.piece_id,
            det_idx=det.det_index, cost=cost,
            cx0=old_cx, cx1=track.center_x, cy0=old_cy, cy1=track.center_y,
            r0=old_r, r1=track.radius, tan0=old_tan, tan1=track.tangent_x,
            row=track.row_id,
        )

    # ---------- logging ----------

    def _log(self, prefix: str, *, level: str, **kw) -> None:
        if self._logger is None:
            return
        if prefix == "PIECE-CREATE":
            msg = (
                "[PIECE] PIECE-CREATE frame={} piece={} cx={:.1f} cy={:.1f} r={:.1f} "
                "tan={:.1f} det_idx={}"
            )
            args = (kw["frame"], kw["piece"], kw["cx"], kw["cy"], kw["r"], kw["tan"], kw["det_idx"])
        elif prefix == "PIECE-MATCH":
            msg = (
                "[PIECE] PIECE-MATCH frame={} piece={} det_idx={} cost={:.2f} "
                "cx={:.1f}->{:.1f} cy={:.1f}->{:.1f} r={:.1f}->{:.1f} tan={:.1f}->{:.1f} row={}"
            )
            args = (
                kw["frame"], kw["piece"], kw["det_idx"], kw["cost"],
                kw["cx0"], kw["cx1"], kw["cy0"], kw["cy1"],
                kw["r0"], kw["r1"], kw["tan0"], kw["tan1"], kw["row"],
            )
        elif prefix == "PIECE-MISS":
            msg = "[PIECE] PIECE-MISS frame={} piece={} missed={} last_cx={:.1f} last_cy={:.1f} row={}"
            args = (kw["frame"], kw["piece"], kw["missed"], kw["last_cx"], kw["last_cy"], kw["row"])
        elif prefix == "PIECE-DROP":
            msg = "[PIECE] PIECE-DROP frame={} piece={} missed={} reason={} row={}"
            args = (kw["frame"], kw["piece"], kw["missed"], kw["reason"], kw["row"])
        elif prefix == "PIECE-AMBIGUOUS-MATCH":
            msg = (
                "[PIECE] PIECE-AMBIGUOUS-MATCH frame={} det_idx={} candidates={} "
                "costs={} chosen={} margin={:.2f}"
            )
            args = (
                kw["frame"], kw["det_idx"], kw["candidates"], kw["costs"],
                kw["chosen"], kw["margin"],
            )
        else:  # pragma: no cover - defensive
            return
        getattr(self._logger, level)(msg, *args)
