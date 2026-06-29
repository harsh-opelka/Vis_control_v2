"""Shared enums + dataclasses passed between the detection thread, the state
machine, and the UI.

These are deliberately small and signal-friendly so we can shuttle them across
Qt's queued connections without ownership headaches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(str, Enum):
    """Top-level UI/control state."""

    WAITING = "WAITING"
    TRACKING = "TRACKING"
    INSPECTING = "INSPECTING"
    READY = "READY"
    FAULT = "FAULT"
    SERVICE = "SERVICE"


class RowPhase(str, Enum):
    """Lifecycle of the row currently at the transfer line.

    IDLE         — no row at line; tripwire armed and watching.
    AT_LINE      — new row occupies line; StopTuchabzug fired; waiting for PLC to stop cloth.
    TRANSFERRING — PLC restarted cloth to hand row to belt; tripwire suppressed for this row.
    """

    IDLE = "IDLE"
    AT_LINE = "AT_LINE"
    TRANSFERRING = "TRANSFERRING"


class Mode(str, Enum):
    """App mode. Selects whether OPC UA is active and whether sim buttons show."""

    DEMO = "demo"
    PRODUCTION = "production"


class Verdict(str, Enum):
    """Outcome of inspecting the belt ROI.

    INFO_* verdicts are informational only — they're logged but do not raise a
    fault. INFO_INCOMPLETE_ROW is the noisiest, so it's logged at DEBUG and not
    appended to the recent-defects list.
    """

    OK = "OK"
    FAULT_ROW_FUSED = "FAULT_ROW_FUSED"
    FAULT_UNKNOWN = "FAULT_UNKNOWN"
    INFO_COLUMN_FUSED = "INFO_COLUMN_FUSED"
    INFO_INCOMPLETE_ROW = "INFO_INCOMPLETE_ROW"
    # FIX 6 (config inspection.unknown_is_fault=False, the default): a blob
    # _classify() couldn't confidently label single/fused that, on a row/column
    # orientation read, is NOT wide/row-wise. Informational only — never a
    # hard fault. FAULT_UNKNOWN is still produced when unknown_is_fault=True
    # (legacy behavior) or when the orientation read says row-wise.
    INFO_UNKNOWN_NONFAULT = "INFO_UNKNOWN_NONFAULT"

    @property
    def is_fault(self) -> bool:
        return self in (Verdict.FAULT_ROW_FUSED, Verdict.FAULT_UNKNOWN)


@dataclass(frozen=True)
class PlcSignals:
    """Snapshot of the 3-variable OPC UA contract.

    - ``tuchabzug_running``: PLC -> us. True while the cloth puller is active.
    - ``stop_tuchabzug``: us -> PLC. We assert this when we detect a row at the
      transfer line; the PLC stops the puller in response.
    - ``fault_active``: us -> PLC. Raised when a fault is detected; cleared
      automatically when the belt has been clean for ``fault_clear_frames``.
    """

    tuchabzug_running: bool = False
    stop_tuchabzug: bool = False
    fault_active: bool = False


@dataclass
class PipelineResult:
    """Result of one detection pass over the current frame.

    ``annotated_image`` is None on the cloth-tracking path (we only render
    annotations for the belt inspection result); the live preview is drawn by
    the UI from the raw frame + detection list.

    ``crop_rect`` is the active detection window in ROI-local coordinates
    (x1, y1, x2, y2).  The camera view widget uses this to dim the area
    outside the configured crop.  None means no crop was applied.
    """

    verdict: Verdict
    detections: list = field(default_factory=list)  # list[Detection]
    inference_ms: float = 0.0
    annotated_image: Optional[object] = None  # numpy ndarray, kept generic to avoid hard dep
    fault_reason: str = ""
    crop_rect: Optional[tuple] = None  # (x1, y1, x2, y2) in ROI-local pixels

    def is_fault(self) -> bool:
        return self.verdict.is_fault
