"""Tests for ``viscontrol.core.event_log``."""

from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from viscontrol.core.event_log import Event, EventLog, EventType


def test_append_creates_file_with_header(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append(
        Event(
            event_type=EventType.STARTUP,
            profile_name="Default",
            extra={"reason": "clean_shutdown"},
        )
    )
    today = log.today_path()
    assert today.exists()

    with today.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["event_type"] == "startup"
    assert rows[0]["profile_name"] == "Default"
    assert "clean_shutdown" in rows[0]["extra_json"]


def test_append_appends_no_duplicate_header(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append(Event(event_type=EventType.PULSE_COMPLETE))
    log.append(Event(event_type=EventType.PULSE_COMPLETE))
    with log.today_path().open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("timestamp_iso")
    assert len(lines) == 3  # header + 2 rows


def test_state_before_and_after_fields(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append(
        Event(
            event_type=EventType.FAULT_RAISED,
            state_before="INSPECTING",
            state_after="FAULT",
            fault_reason="row_fused",
            image_filename="logs/defects/2026-05-21-12-00-00.jpg",
        )
    )
    with log.today_path().open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["state_before"] == "INSPECTING"
    assert row["state_after"] == "FAULT"
    assert row["fault_reason"] == "row_fused"
    assert row["image_filename"].endswith(".jpg")


def test_daily_rollover(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    yesterday = datetime.now() - timedelta(days=1)
    log.append(Event(event_type=EventType.STARTUP, timestamp=yesterday))
    log.append(Event(event_type=EventType.PULSE_COMPLETE))  # today
    files = sorted(p.name for p in tmp_path.glob("events-*.csv"))
    assert len(files) == 2


def test_prune_older_than(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    old = tmp_path / "events-2020-01-01.csv"
    old.write_text("timestamp_iso\n", encoding="utf-8")
    # Backdate mtime well past the cutoff.
    past = time.time() - 100 * 86400
    import os
    os.utime(old, (past, past))
    log.append(Event(event_type=EventType.STARTUP))  # creates today's file

    removed = log.prune_older_than(days=30)
    assert removed == 1
    assert not old.exists()
    assert log.today_path().exists()


def test_prune_zero_days_rejected(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    with pytest.raises(ValueError):
        log.prune_older_than(days=0)
