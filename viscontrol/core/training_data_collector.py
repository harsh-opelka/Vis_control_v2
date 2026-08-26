"""Training-data image collection — additive, alongside error-image saving.

Curates a bounded, compressed sample of frames for future YOLO/CV model
training: rare/hard cases (layout-lock faults, ambiguous row associations)
are always kept; everything else is sparsely sampled. A rolling cleanup
(:meth:`TrainingDataCollector.run_maintenance`) enforces a total on-disk
size cap by removing whole oldest date-folders, so an unattended Jetson
never fills its SD card.

No Qt. cv2 + stdlib only — this module must stay importable and testable
without a display or camera. QTimer/QThread wiring for
:meth:`run_maintenance` lives in viscontrol/ui/main_window.py.

This module does NOT touch, call, or replace the existing Einlaufband
error-image saving path (MainWindow._save_defect_image /
storage.defect_image_dir) — it is a separate, independent output tree
under cfg.output_dir.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from viscontrol.core.logger import logger as _default_logger


@dataclass(frozen=True)
class SaveDecision:
    should_save: bool
    category: str  # "normal_sample" | "ambiguous" | "layout_fault"
    reason: str


class TrainingDataCollector:
    """Decides which row-event frames to keep and writes them, compressed.

    Pure side-effect observer: every public method catches and logs its own
    failures rather than raising, so a training-data problem (disk full,
    bad path, encode failure) can never interrupt production detection.
    """

    def __init__(self, cfg, logger=None) -> None:
        self._cfg = cfg
        self._logger = logger if logger is not None else _default_logger
        self._output_dir = Path(cfg.output_dir)
        self._last_low_disk_warning_ts: float = 0.0

    # ---------- sampling decision ----------

    def decide(
        self,
        frame_id: int,
        is_ambiguous: bool,
        layout_fault_active: bool,
    ) -> SaveDecision:
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
        elif frame_id % max(1, self._cfg.normal_sample_every_n_rows) == 0:
            decision = SaveDecision(
                should_save=True, category="normal_sample",
                reason=f"row_event_mod_{self._cfg.normal_sample_every_n_rows}",
            )
        else:
            decision = SaveDecision(
                should_save=False, category="normal_sample", reason="not_sampled",
            )

        if decision.should_save:
            self._logger.info(
                "[TRAINDATA] SAVE-DECISION frame={} category={} reason={}",
                frame_id, decision.category, decision.reason,
            )
        return decision

    # ---------- save (disk-guarded, compressed) ----------

    def save(self, image: np.ndarray, decision: SaveDecision, frame_id: int) -> str | None:
        if not decision.should_save:
            return None
        try:
            free_gb = shutil.disk_usage(_nearest_existing(self._output_dir)).free / (1024 ** 3)
        except OSError:
            # Can't stat the target filesystem — fail closed (don't save)
            # rather than risk writing onto an unknown/missing volume.
            self._logger.exception("[TRAINDATA-ERROR] disk_usage check failed")
            return None

        if free_gb < self._cfg.min_free_space_gb:
            now = time.monotonic()
            if now - self._last_low_disk_warning_ts >= self._cfg.low_disk_warning_cooldown_s:
                self._last_low_disk_warning_ts = now
                self._logger.warning(
                    "[TRAINDATA] SAVE-BLOCKED-LOW-DISK free_gb={:.1f} threshold_gb={}",
                    free_gb, self._cfg.min_free_space_gb,
                )
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
