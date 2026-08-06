"""JSONL recording of fresh detection updates, for offline replay.

One line per FRESH detection update (see tests/test_replay.py:
R3 recorded_log_replay, which replays a recording captured this way through
PieceTracker -> RowTracker -> TransferOrchestrator). No Qt. No cv2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordedFrame:
    frame_id: int
    ts: float
    transfer_x: float
    is_fresh: bool
    cycle_id: int
    detections: list  # list[dict] with center_x/center_y/radius keys


class DetectionRecorder:
    """Appends one JSON line per fresh detection update to ``path``.

    A no-op when ``enabled`` is False (the common case — see
    config/default.yaml: detection_recorder.enabled) so callers can
    unconditionally call :meth:`record` every frame with zero overhead.
    """

    def __init__(self, path, enabled: bool) -> None:
        self._enabled = enabled
        self._path = Path(path) if path is not None else None
        self._fh = None
        if self._enabled and self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")

    def record(
        self, frame_id: int, timestamp: float, transfer_x: float, detections: list,
        is_fresh: bool, *, cycle_id: int = 0,
    ) -> None:
        if not self._enabled or self._fh is None or not is_fresh:
            return
        line = {
            "frame_id": frame_id,
            "ts": timestamp,
            "transfer_x": transfer_x,
            "is_fresh": is_fresh,
            "cycle_id": cycle_id,
            "detections": [
                {"center_x": d.center_x, "center_y": d.center_y, "radius": d.radius}
                for d in detections
            ],
        }
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def load_recording(path) -> list:
    """Load a JSONL recording written by :class:`DetectionRecorder`.

    Returns ``list[RecordedFrame]`` in file order.
    """
    frames: list[RecordedFrame] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            obj = json.loads(raw_line)
            frames.append(RecordedFrame(
                frame_id=obj["frame_id"],
                ts=obj["ts"],
                transfer_x=obj["transfer_x"],
                is_fresh=obj["is_fresh"],
                cycle_id=obj.get("cycle_id", 0),
                detections=obj["detections"],
            ))
    return frames
