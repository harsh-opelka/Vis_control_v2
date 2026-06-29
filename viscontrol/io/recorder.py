"""Raw-frame recorder for offline calibration.

CALIBRATION TOOLING — not part of the detection/tripwire/state-machine path.
:class:`FrameRecorder` only observes frames already produced by the camera
and writes them, unmodified, to lossless PNG files on a background thread.
Recording or not recording has zero effect on detection behavior.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from viscontrol.core.logger import logger


class FrameRecorder:
    """Background-thread PNG writer for raw frame capture.

    ``submit()`` is non-blocking: if the write queue is full (the writer
    can't keep up with PNG encoding) the frame is dropped and counted rather
    than blocking the caller — the camera/UI thread must never stall on disk
    I/O.
    """

    def __init__(self, output_root: Path, *, max_queue: int = 8) -> None:
        self._output_root = Path(output_root)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._saved_count = 0
        self._dropped_count = 0
        self._session_dir: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def saved_count(self) -> int:
        with self._lock:
            return self._saved_count

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def start(self) -> Path:
        """Begin a new recording session into ``output_root/<timestamp>/``."""
        if self.is_recording:
            raise RuntimeError("FrameRecorder already running")
        self._session_dir = self._output_root / time.strftime("%Y%m%d_%H%M%S")
        self._session_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._saved_count = 0
            self._dropped_count = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="FrameRecorder", daemon=True)
        self._thread.start()
        logger.info("FrameRecorder started: {}", self._session_dir)
        return self._session_dir

    def submit(self, frame: np.ndarray) -> None:
        """Queue *frame* (copied) for writing. Drops it if the queue is full."""
        if not self.is_recording:
            return
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            with self._lock:
                self._dropped_count += 1

    def stop(self) -> None:
        if not self.is_recording:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        logger.info(
            "FrameRecorder stopped: {} saved, {} dropped, dir={}",
            self.saved_count, self.dropped_count, self._session_dir,
        )

    def _run(self) -> None:
        idx = 0
        # Drain the queue after stop() is requested so the tail of the
        # recording isn't lost to the backlog.
        while not self._stop.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            idx += 1
            path = self._session_dir / f"frame_{idx:06d}.png"
            try:
                # Compression level 1: still lossless, just fast to keep up
                # with capture rate. Pixel values are preserved exactly.
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                with self._lock:
                    self._saved_count += 1
            except Exception:  # noqa: BLE001
                logger.exception("FrameRecorder failed to write {}", path)