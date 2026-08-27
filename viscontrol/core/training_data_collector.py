"""Training-data image collection — additive, alongside error-image saving.

Curates a bounded, compressed sample of frames for future YOLO/CV model
training. Three independent capture paths:

1. Row-lifecycle capture (:meth:`on_row_tripwire`): exactly one image
   per selected row_id, at the "tripwire" stage (ACTIVE->STOP_REQUESTED
   — only for rows that actually get a stop command). Each saved image
   gets a JSON metadata sidecar (see :meth:`_save_row_capture`). (The
   previous "confirmed" DETECTED->ACTIVE stage was removed earlier — see
   dropconfirmed_inspection.txt — and the "cleared" terminal-transition
   stage has since been removed too — see dropcleared_inspection.txt;
   "tripwire" is now the only row-lifecycle stage.) Subject to a "save
   every Nth row" sample filter (:meth:`_should_capture_row`,
   cfg.capture_every_n_rows) — see nth_row_inspection.txt: only every
   Nth distinct row_id this collector observes is actually saved.
2. Hard-case capture (:meth:`decide` / :meth:`save`, unchanged from the
   previous version): layout-lock faults and ambiguous row associations
   are always saved, regardless of row-lifecycle state — a separate,
   still-valid path for hard cases outside normal row flow.
3. Einlaufband fault capture (:meth:`on_einlaufband_fault`): an
   ADDITIONAL, additive save into this same output tree (category
   "fault") every time the existing, untouched Einlaufband fault-save
   path (MainWindow._save_defect_image) fires — see
   dropconfirmed_inspection.txt part B. Never replaces or affects that
   existing save in any way.

A rolling cleanup (:meth:`TrainingDataCollector.run_maintenance`) enforces
a total on-disk size cap by removing whole oldest date-folders, so an
unattended Jetson never fills its SD card.

No Qt. cv2 + stdlib only — this module must stay importable and testable
without a display or camera. QTimer/QThread wiring for
:meth:`run_maintenance` lives in viscontrol/ui/main_window.py.

This module does NOT touch, call, or replace the existing Einlaufband
error-image saving path (MainWindow._save_defect_image /
storage.defect_image_dir) — it is a separate, independent output tree
under cfg.output_dir.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from viscontrol.core.logger import logger as _default_logger


@dataclass(frozen=True)
class SaveDecision:
    should_save: bool
    category: str  # "ambiguous" | "layout_fault" | "none"
    reason: str


@dataclass(frozen=True)
class RowPieceData:
    """Real, already-computed row data handed in by the caller (MainWindow)
    at capture time — see row_capture_inspection.txt part C for exactly
    which fields exist on RowRecord and which do not.

    ``tangent_xs`` is the row's (front_tangent, back_tangent) aggregate
    pair — the only per-row "position" data the orchestrator retains.
    Individual per-piece tangent_x values are not available at the
    row/orchestrator level (they don't propagate past
    proximity_clustering.cluster_by_tangent), so this is NOT a per-piece
    list — never fabricated to look like one.

    ``fused`` is always None: there is no row_fused/column_fused concept
    anywhere in the cloth-side row-lifecycle data model (RowRecord /
    TransferEvent / ClusterObservation) — the only "fused" concept in the
    codebase is the belt-side Verdict.FAULT_ROW_FUSED, which has no
    mapping back to a cloth row_id. Kept as a field (rather than removed)
    so the metadata schema has a stable place for it if that wiring is
    ever added later; always null until then.
    """

    piece_count: int
    tangent_xs: tuple
    fused: bool | None = None


@dataclass(frozen=True)
class RowCaptureEvent:
    row_id: int
    cycle_id: int
    stage: str  # "tripwire"
    fused: bool | None  # None if not yet known at this stage
    piece_count: int
    tangent_xs: tuple


class TrainingDataCollector:
    """Decides which frames to keep and writes them, compressed.

    Pure side-effect observer: every public method catches and logs its own
    failures rather than raising, so a training-data problem (disk full,
    bad path, encode failure) can never interrupt production detection.
    """

    _STAGES = ("tripwire",)

    def __init__(self, cfg, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger if logger is not None else _default_logger
        self._output_dir = Path(cfg.output_dir)
        self._last_low_disk_warning_ts: float = 0.0

        # Per-row dedup guard: row_id -> set of stages already captured.
        # Bounded to cfg.max_tracked_row_ids distinct row_ids so a long
        # production run can never grow this unboundedly — oldest row_id
        # evicted (FIFO) once the bound is exceeded.
        self._captured_stages: dict[int, set[str]] = {}
        self._row_id_eviction_order: deque[int] = deque()

        # "Save every Nth row" filter (see nth_row_inspection.txt) — applies
        # ONLY to the row-lifecycle capture (on_row_tripwire) below, never
        # to the ambiguous/layout_fault or Einlaufband-fault paths.
        # Self-contained: counts distinct row_ids THIS collector has
        # observed (never resets, independent of any orchestrator cycle
        # reset) rather than reading the orchestrator's row_id value — see
        # nth_row_inspection.txt part B for why. (No per-row-id decision
        # cache is needed: "tripwire" is the only row-lifecycle stage —
        # see dropcleared_inspection.txt — and _mark_if_new above already
        # guarantees this method is reached at most once per row_id, so
        # there is no second call for it to agree with.)
        self._rows_seen_count: int = 0

    # ---------- hard-case decision (ambiguous / layout_fault only) ----------

    def decide(
        self,
        frame_id: int,
        is_ambiguous: bool,
        layout_fault_active: bool,
    ) -> SaveDecision:
        """Layout-lock faults and ambiguous row associations are always
        saved — a separate, still-valid capture path for hard cases,
        independent of the row-lifecycle capture below."""
        if layout_fault_active:
            decision = SaveDecision(
                should_save=True, category="layout_fault",
                reason="layout_lock_active",
            )
        elif is_ambiguous:
            decision = SaveDecision(
                should_save=True, category="ambiguous",
                reason="event_ambiguous",
            )
        else:
            decision = SaveDecision(
                should_save=False, category="none", reason="not_ambiguous_or_fault",
            )

        if decision.should_save:
            self._logger.info(
                "[TRAINDATA] SAVE-DECISION frame={} category={} reason={}",
                frame_id, decision.category, decision.reason,
            )
        return decision

    def save(self, image: np.ndarray, decision: SaveDecision, frame_id: int) -> str | None:
        if not decision.should_save:
            return None
        if not self._disk_ok():
            return None
        try:
            ok, buf = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality],
            )
            if not ok:
                self._logger.error("[TRAINDATA-ERROR] jpeg encode failed frame={}", frame_id)
                return None

            date_str = datetime.now().strftime("%Y-%m-%d")
            out_dir = self._output_dir / decision.category / date_str
            out_dir.mkdir(parents=True, exist_ok=True)

            stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            path = out_dir / f"{frame_id}_{stamp}.jpg"
            buf.tofile(str(path))
            return str(path)
        except Exception:  # noqa: BLE001
            self._logger.exception(
                "[TRAINDATA-ERROR] failed to save training image frame={}", frame_id,
            )
            return None

    # ---------- row-lifecycle capture (1 explicit stage per row_id) ----------

    def on_row_tripwire(
        self, row_id: int, cycle_id: int, image: np.ndarray | None, piece_data: RowPieceData,
    ) -> None:
        """ACTIVE -> STOP_REQUESTED — only fires for rows that actually get
        a stop command (see row_capture_inspection.txt part A.2). The only
        row-lifecycle capture stage — see dropcleared_inspection.txt for
        why the "cleared" terminal-transition stage was removed."""
        self._capture(row_id, cycle_id, "tripwire", image, piece_data)

    def _capture(
        self,
        row_id: int,
        cycle_id: int,
        stage: str,
        image: np.ndarray | None,
        piece_data: RowPieceData,
    ) -> None:
        try:
            if not self._mark_if_new(row_id, stage):
                self._logger.info(
                    "[TRAINDATA] ROW-CAPTURE-SKIPPED-DUPLICATE row={} stage={}",
                    row_id, stage,
                )
                return

            if not self._should_capture_row():
                self._logger.info(
                    "[TRAINDATA] ROW-SKIPPED-BY-SAMPLE row={} rows_seen={} every_n={}",
                    row_id, self._rows_seen_count, self._cfg.capture_every_n_rows,
                )
                return

            event = RowCaptureEvent(
                row_id=row_id, cycle_id=cycle_id, stage=stage,
                fused=piece_data.fused, piece_count=piece_data.piece_count,
                tangent_xs=tuple(piece_data.tangent_xs),
            )
            self._logger.info(
                "[TRAINDATA] ROW-CAPTURE row={} cycle={} stage={} fused={}",
                row_id, cycle_id, stage, event.fused,
            )
            if image is None:
                return
            self._save_row_capture(image, event)
        except Exception:  # noqa: BLE001
            self._logger.exception(
                "[TRAINDATA-ERROR] row capture failed row={} stage={}", row_id, stage,
            )

    def _mark_if_new(self, row_id: int, stage: str) -> bool:
        """Dedup guard: True (and marks captured) the first time (row_id,
        stage) is seen; False on any repeat. Bounded to
        cfg.max_tracked_row_ids distinct row_ids — oldest evicted FIFO."""
        stages = self._captured_stages.get(row_id)
        if stages is None:
            stages = set()
            self._captured_stages[row_id] = stages
            self._row_id_eviction_order.append(row_id)
            bound = max(1, self._cfg.max_tracked_row_ids)
            while len(self._row_id_eviction_order) > bound:
                oldest = self._row_id_eviction_order.popleft()
                self._captured_stages.pop(oldest, None)
        if stage in stages:
            return False
        stages.add(stage)
        return True

    def _should_capture_row(self) -> bool:
        """Save-every-Nth-row filter — applies only to row-lifecycle
        capture (never ambiguous/layout_fault/fault). "tripwire" is the
        only row-lifecycle stage (see dropcleared_inspection.txt), and
        `_mark_if_new` (called first in `_capture`) already guarantees
        this is reached at most once per row_id, so a straight increment-
        and-modulo check is sufficient — no per-row-id decision cache is
        needed any more.
        """
        self._rows_seen_count += 1
        every_n = max(1, self._cfg.capture_every_n_rows)
        return self._rows_seen_count % every_n == 0

    def _save_row_capture(self, image: np.ndarray, event: RowCaptureEvent) -> str | None:
        if not self._disk_ok():
            return None
        try:
            ok, buf = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality],
            )
            if not ok:
                self._logger.error(
                    "[TRAINDATA-ERROR] jpeg encode failed row={} stage={}",
                    event.row_id, event.stage,
                )
                return None

            date_str = datetime.now().strftime("%Y-%m-%d")
            out_dir = self._output_dir / event.stage / date_str
            out_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now()
            stamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
            base = f"{event.row_id}_{event.cycle_id}_{event.stage}_{stamp}"

            img_path = out_dir / f"{base}.jpg"
            buf.tofile(str(img_path))

            metadata = {
                "row_id": event.row_id,
                "cycle_id": event.cycle_id,
                "stage": event.stage,
                "timestamp": now.isoformat(),
                "fused": event.fused,
                "piece_count": event.piece_count,
                "tangent_xs": list(event.tangent_xs),
            }
            json_path = out_dir / f"{base}.json"
            json_path.write_text(json.dumps(metadata), encoding="utf-8")
            return str(img_path)
        except Exception:  # noqa: BLE001
            self._logger.exception(
                "[TRAINDATA-ERROR] failed to save row capture row={} stage={}",
                event.row_id, event.stage,
            )
            return None

    # ---------- Einlaufband fault capture (additive — see dropconfirmed_inspection.txt part B) ----------

    def on_einlaufband_fault(
        self, image: np.ndarray | None, error_code: int, metadata: dict | None = None,
    ) -> None:
        """Additional save into training_data/fault/{date}/, alongside
        (never instead of) the existing, untouched
        MainWindow._save_defect_image / storage.defect_image_dir path.
        Same jpeg_quality/disk-guard/cleanup as every other category —
        just a 4th category folder. ``metadata`` carries any extra fields
        trivially available at the caller's fault-save call site (e.g.
        verdict, fault_reason); passed through as-is, never invented —
        the three sidecar fields below always win over anything in
        ``metadata`` with the same key.
        """
        try:
            if image is None:
                return
            if not self._disk_ok():
                return
            ok, buf = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality],
            )
            if not ok:
                self._logger.error(
                    "[TRAINDATA-ERROR] jpeg encode failed category=fault error_code={}",
                    error_code,
                )
                return

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            out_dir = self._output_dir / "fault" / date_str
            out_dir.mkdir(parents=True, exist_ok=True)

            stamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
            base = f"fault_{error_code}_{stamp}"

            img_path = out_dir / f"{base}.jpg"
            buf.tofile(str(img_path))

            sidecar: dict = dict(metadata) if metadata else {}
            sidecar["category"] = "fault"
            sidecar["error_code"] = error_code
            sidecar["timestamp"] = now.isoformat()
            json_path = out_dir / f"{base}.json"
            json_path.write_text(json.dumps(sidecar), encoding="utf-8")

            self._logger.info(
                "[TRAINDATA] FAULT-CAPTURE error_code={} path={}", error_code, img_path,
            )
        except Exception:  # noqa: BLE001
            self._logger.exception(
                "[TRAINDATA-ERROR] on_einlaufband_fault capture failed error_code={}",
                error_code,
            )

    # ---------- disk guard (shared by all capture paths) ----------

    def _disk_ok(self) -> bool:
        try:
            free_gb = shutil.disk_usage(_nearest_existing(self._output_dir)).free / (1024 ** 3)
        except OSError:
            # Can't stat the target filesystem — fail closed (don't save)
            # rather than risk writing onto an unknown/missing volume.
            self._logger.exception("[TRAINDATA-ERROR] disk_usage check failed")
            return False

        if free_gb < self._cfg.min_free_space_gb:
            now = time.monotonic()
            if now - self._last_low_disk_warning_ts >= self._cfg.low_disk_warning_cooldown_s:
                self._last_low_disk_warning_ts = now
                self._logger.warning(
                    "[TRAINDATA] SAVE-BLOCKED-LOW-DISK free_gb={:.1f} threshold_gb={}",
                    free_gb, self._cfg.min_free_space_gb,
                )
            return False
        return True

    # ---------- rolling cleanup ----------

    def run_maintenance(self) -> None:
        """Enforce cfg.max_total_size_gb by deleting whole oldest date-folders.

        Cheap and deterministic: folder age comes from the YYYY-MM-DD name,
        not per-file mtime, so no per-file stat pass is needed to pick
        deletion order (each folder's own size still requires walking its
        files once, to know how many bytes deleting it frees).

        Never raises — called from a background thread on a timer; any
        failure here must not propagate.
        """
        try:
            self._run_maintenance_impl()
        except Exception:  # noqa: BLE001
            self._logger.exception("[TRAINDATA-ERROR] run_maintenance failed")

    def _run_maintenance_impl(self) -> None:
        if not self._output_dir.exists():
            self._logger.info(
                "[TRAINDATA] CLEANUP-SUMMARY total_size_gb={:.2f} limit_gb={} "
                "folders_removed={} categories={{}}",
                0.0, self._cfg.max_total_size_gb, 0,
            )
            return

        # date_str -> {category -> size_bytes}
        date_category_sizes: dict[str, dict[str, int]] = {}
        for category_dir in self._output_dir.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for date_dir in category_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                size = _dir_size_bytes(date_dir)
                date_category_sizes.setdefault(date_dir.name, {})[category] = size

        total_bytes = sum(
            size for cats in date_category_sizes.values() for size in cats.values()
        )
        limit_bytes = self._cfg.max_total_size_gb * (1024 ** 3)

        folders_removed = 0
        removed_category_counts: dict[str, int] = {}

        if total_bytes > limit_bytes:
            # Oldest date string first — deterministic, no file mtime needed.
            for date_str in sorted(date_category_sizes.keys()):
                if total_bytes <= limit_bytes:
                    break
                for category, size in list(date_category_sizes[date_str].items()):
                    folder = self._output_dir / category / date_str
                    freed_gb = size / (1024 ** 3)
                    try:
                        shutil.rmtree(folder)
                    except OSError:
                        self._logger.exception(
                            "[TRAINDATA-ERROR] failed to remove {}", folder,
                        )
                        continue
                    total_bytes -= size
                    folders_removed += 1
                    removed_category_counts[category] = (
                        removed_category_counts.get(category, 0) + 1
                    )
                    self._logger.info(
                        "[TRAINDATA] CLEANUP-REMOVE folder={} freed_estimate_gb={:.2f}",
                        folder, freed_gb,
                    )
                del date_category_sizes[date_str]

        # Per-category folder counts remaining, for the summary log.
        remaining_category_counts: dict[str, int] = {}
        for cats in date_category_sizes.values():
            for category in cats:
                remaining_category_counts[category] = (
                    remaining_category_counts.get(category, 0) + 1
                )

        self._logger.info(
            "[TRAINDATA] CLEANUP-SUMMARY total_size_gb={:.2f} limit_gb={} "
            "folders_removed={} categories={}",
            total_bytes / (1024 ** 3), self._cfg.max_total_size_gb,
            folders_removed, remaining_category_counts,
        )


def _nearest_existing(path: Path) -> Path:
    """Walk up to the nearest existing ancestor (for disk_usage on a
    not-yet-created output_dir); falls back to the path's drive/anchor."""
    p = path
    while not p.exists():
        parent = p.parent
        if parent == p:
            return p
        p = parent
    return p


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total
