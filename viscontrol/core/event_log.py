"""CSV-based event log, daily-rolled.

A small, append-only audit trail of state transitions, fault events,
mode/profile/language changes, and lifecycle events. The format is fixed for
forward compatibility with any downstream reporting tools the line operator
might wire up later.
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    """Closed set of event types — enforced so dashboards can rely on the schema."""

    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    FAULT_RAISED = "fault_raised"
    FAULT_CLEARED = "fault_cleared"
    PULSE_COMPLETE = "pulse_complete"
    PROFILE_CHANGE = "profile_change"
    CALIBRATION_DONE = "calibration_done"
    MODE_CHANGE = "mode_change"
    WIZARD_COMPLETE = "wizard_complete"
    CAMERA_DISCONNECTED = "camera_disconnected"
    CAMERA_RECONNECTED = "camera_reconnected"


_COLUMNS = [
    "timestamp_iso",
    "event_type",
    "profile_name",
    "fault_reason",
    "image_filename",
    "state_before",
    "state_after",
    "extra_json",
]


@dataclass
class Event:
    """A single row in the daily CSV.

    ``extra`` is a free-form dict serialized to ``extra_json`` so callers can
    add fields without changing the schema.
    """

    event_type: EventType
    profile_name: str = ""
    fault_reason: str = ""
    image_filename: str = ""
    state_before: str = ""
    state_after: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None


class EventLog:
    """Thread-safe daily-rolled CSV event log.

    Writes are flushed after every row so a sudden power loss still leaves a
    readable file. Filenames roll at local-midnight (``events-YYYY-MM-DD.csv``).
    """

    def __init__(self, log_dir: Path) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, ts: datetime) -> Path:
        return self._dir / f"events-{ts.strftime('%Y-%m-%d')}.csv"

    def append(self, event: Event) -> Path:
        """Append a row, returning the file path used."""
        ts = event.timestamp or datetime.now()
        path = self._path_for(ts)
        row = {
            "timestamp_iso": ts.isoformat(timespec="milliseconds"),
            "event_type": event.event_type.value,
            "profile_name": event.profile_name,
            "fault_reason": event.fault_reason,
            "image_filename": event.image_filename,
            "state_before": event.state_before,
            "state_after": event.state_after,
            "extra_json": json.dumps(event.extra, separators=(",", ":")) if event.extra else "",
        }
        with self._lock:
            new_file = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
        return path

    def today_path(self) -> Path:
        """Path of the CSV the next append would target. Useful for the web sidecar."""
        return self._path_for(datetime.now())

    def prune_older_than(self, days: int) -> int:
        """Delete event CSVs older than ``days``. Returns the number removed."""
        if days < 1:
            raise ValueError("days must be >= 1")
        cutoff = datetime.now().timestamp() - days * 86400
        removed = 0
        for path in self._dir.glob("events-*.csv"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed
