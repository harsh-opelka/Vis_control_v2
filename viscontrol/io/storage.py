"""File-system helpers for runtime artifacts.

Currently:
- :func:`save_defect_image` — writes a defect frame under
  ``storage.defect_image_dir`` with a timestamped name.
- :func:`prune_defect_images` — keeps only the most recent N days of files.
- :func:`mark_clean_shutdown` / :func:`detect_unexpected_reboot` —
  drop / read a small marker file in the log dir so we can tell reboots apart.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from viscontrol.core.logger import logger

_SHUTDOWN_MARKER_NAME = "shutdown.marker"


def save_defect_image(
    image: np.ndarray, defect_dir: Path, *, suffix: str = ""
) -> Path | None:
    defect_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    name = f"{stamp}{('-' + suffix) if suffix else ''}.jpg"
    path = defect_dir / name
    try:
        if not cv2.imwrite(str(path), image):
            return None
    except Exception:  # noqa: BLE001
        logger.exception("failed to save defect image {}", path)
        return None
    return path


def prune_defect_images(defect_dir: Path, *, keep_days: int) -> int:
    if keep_days < 1:
        raise ValueError("keep_days must be >= 1")
    if not defect_dir.exists():
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    for path in defect_dir.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def mark_clean_shutdown(log_dir: Path) -> None:
    """Write a marker that indicates the last shutdown was clean.

    Called from main.py on QApplication.aboutToQuit. If the file is *missing*
    on next startup we can presume an unexpected reboot.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / _SHUTDOWN_MARKER_NAME).write_text(
        datetime.now().isoformat(timespec="seconds"),
        encoding="utf-8",
    )


def detect_unexpected_reboot(log_dir: Path) -> tuple[bool, str]:
    """Return (was_unexpected, reason_label).

    ``True, "unexpected_reboot"`` if the marker is missing on startup; we then
    consume (delete) the marker if present so the *next* unclean exit can be
    detected too.
    """
    marker = log_dir / _SHUTDOWN_MARKER_NAME
    if marker.exists():
        try:
            marker.unlink()
        except OSError:
            pass
        return False, "clean_shutdown"
    return True, "unexpected_reboot"
