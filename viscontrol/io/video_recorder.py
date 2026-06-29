"""Background-thread MP4 recorder for the processed inference view.

REVIEW/OBSERVABILITY TOOLING — not part of the detection / tripwire / row-grouping
/ state-machine / OPC UA path. :class:`ViewVideoRecorder` only receives frames
the UI has already rendered (the inference view grabbed as an image, overlays and
all) and encodes them to an ``.mp4`` on a background thread. Whether it is
recording or not has zero effect on detection behaviour.

``submit()`` is non-blocking: if the encode queue is full (the writer can't keep
up with the live grab rate) the frame is dropped and counted rather than blocking
the GUI thread — the production pipeline and UI must never stall on the recorder.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from viscontrol.core.logger import logger


class ViewVideoRecorder:
    """Background-thread MP4 writer for the rendered inference view.

    One ``start()`` → ``stop()`` cycle produces one ``output_root/<timestamp>.mp4``
    file. The frame size is fixed at ``start()``; later frames of a different size
    (e.g. the window was resized) are resized to match so the file stays valid.
    """

    def __init__(
        self,
        output_root: Path,
        *,
        fps: float = 15.0,
        max_queue: int = 16,
    ) -> None:
        self._output_root = Path(output_root)
        self._fps = float(fps)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None  # (width, height)
        self._output_path: Path | None = None
        self._written_count = 0
        self._dropped_count = 0

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def written_count(self) -> int:
        with self._lock:
            return self._written_count

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def start(self, width: int, height: int) -> Path | None:
        """Open a new MP4 sized ``width × height`` and start the writer thread.

        Returns the output path, or ``None`` if the file/codec could not be
        opened (logged) — the caller should then abort recording gracefully.
        """
        if self.is_recording:
            raise RuntimeError("ViewVideoRecorder already running")
        # mp4v: MPEG-4 Part 2, bundled with OpenCV's FFmpeg on all platforms;
        # tolerant of arbitrary dimensions. Force even dims to stay safe.
        w = max(2, int(width) - (int(width) % 2))
        h = max(2, int(height) - (int(height) % 2))
        self._output_root.mkdir(parents=True, exist_ok=True)
        path = self._output_root / (time.strftime("%Y%m%d_%H%M%S") + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            logger.error("ViewVideoRecorder: failed to open writer for {}", path)
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                pass
            return None

        self._writer = writer
        self._size = (w, h)
        self._output_path = path
        with self._lock:
            self._written_count = 0
            self._dropped_count = 0
        # Drain any stale frames from a previous session.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ViewVideoRecorder", daemon=True
        )
        self._thread.start()
        logger.info("ViewVideoRecorder started: {} ({}x{} @ {:.0f}fps)", path, w, h, self._fps)
        return path

    def submit(self, frame: np.ndarray) -> None:
        """Queue *frame* (BGR uint8) for encoding. Drops it if the queue is full."""
        if not self.is_recording:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            with self._lock:
                self._dropped_count += 1

    def stop(self) -> Path | None:
        """Finalise the file. Returns the saved path (or None if not recording)."""
        if not self.is_recording:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        path = self._output_path
        logger.info(
            "ViewVideoRecorder stopped: {} written, {} dropped, file={}",
            self.written_count, self.dropped_count, path,
        )
        return path

    def _run(self) -> None:
        assert self._writer is not None and self._size is not None
        w, h = self._size
        # Keep draining after stop() so the tail of the recording isn't lost.
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    frame = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.shape[1] != w or frame.shape[0] != h:
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    self._writer.write(frame)
                    with self._lock:
                        self._written_count += 1
                except Exception:  # noqa: BLE001
                    logger.exception("ViewVideoRecorder failed to write a frame")
        finally:
            try:
                self._writer.release()
            except Exception:  # noqa: BLE001
                logger.exception("ViewVideoRecorder writer.release failed")
            self._writer = None
