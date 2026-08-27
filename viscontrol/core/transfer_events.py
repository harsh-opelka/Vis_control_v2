"""Pure data model for the transfer/row lifecycle (see transfer_orchestrator.py).

No Qt imports. No OpenCV imports. No I/O. These types are shared between
row_associator.py, transfer_orchestrator.py, and their tests; MainWindow only
ever touches them through TransferOrchestrator's public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RowState(str, Enum):
    DETECTED = "DETECTED"
    ACTIVE = "ACTIVE"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPED = "STOPPED"
    TRANSFER_PENDING = "TRANSFER_PENDING"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    ABANDONED = "ABANDONED"  # terminal; never counts as transferred


class EventType(str, Enum):
    CYCLE_START = "CYCLE_START"
    OBSERVATION = "OBSERVATION"  # one frame of validated clusters
    PLC_STOP_ACK = "PLC_STOP_ACK"  # tuchabzug_running TRUE -> FALSE
    PLC_CLOTH_START = "PLC_CLOTH_START"  # tuchabzug_running FALSE -> TRUE
    CYCLE_END = "CYCLE_END"


@dataclass(frozen=True)
class ClusterObservation:
    """Immutable per-frame observation. Carries NO persistent state."""

    front_tangent: float  # min tangent_x  (leading edge, toward transfer line)
    back_tangent: float  # max tangent_x  (trailing edge)
    piece_count: int
    center_y: float  # logged only; NEVER used for matching
    temp_cluster_index: int  # per-frame index only; NOT identity


@dataclass(frozen=True)
class TransferEvent:
    event_id: int  # globally monotonic, assigned by orchestrator
    cycle_id: int
    event_type: EventType
    timestamp: float  # monotonic seconds
    source_frame_id: int
    observations: tuple = ()  # tuple[ClusterObservation, ...] for OBSERVATION
    row_id: int | None = None  # set when already known


@dataclass(frozen=True)
class TransferSnapshot:
    """Immutable ownership record handed to the PLC/stop path.

    Once created it is never mutated and never reassigned to another row.
    """

    cycle_id: int
    row_id: int
    track_id: int
    stop_event_id: str
    transfer_event_id: str | None
    created_at: float
    created_from_frame_id: int


@dataclass
class RowRecord:
    row_id: int  # PERMANENT identity. Never reused. Never an index.
    cycle_id: int
    track_id: int  # stable association track id
    display_number: int  # presentation only (R1, R2, R3)
    state: RowState
    front_tangent: float
    back_tangent: float
    first_seen_frame: int
    last_seen_frame: int
    confirmed_upstream_frames: int
    missed_frames: int
    piece_count: int = 0  # last known piece count from RowAssociator matches;
                          # read by the training-data row-lifecycle capture
                          # (see viscontrol/core/training_data_collector.py) —
                          # not used by any association/lifecycle decision
    prev_front_tangent: float = 0.0  # snapshotted before each match, for tripwire crossing
    stop_event_id: str | None = None
    transfer_event_id: str | None = None
    snapshot: TransferSnapshot | None = None
    created_at: float = 0.0
    stop_requested_at: float | None = None
    stopped_at: float | None = None
    transfer_started_at: float | None = None
    transferred_at: float | None = None
    history: list = field(default_factory=list)  # list[tuple[float, RowState, RowState, str]]


@dataclass(frozen=True)
class OrchestratorSnapshot:
    """What the UI reads. UI must NEVER mutate orchestrator state."""

    cycle_id: int
    raw_cluster_count: int
    rows: tuple  # tuple[RowRecord, ...] — copies, ordered by display_number
    rows_active: int
    rows_stop_requested: int
    rows_transfer_pending: int
    rows_transferring: int
    rows_transferred: int
    rows_abandoned: int
