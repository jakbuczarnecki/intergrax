# © Artur Czarnecki. All rights reserved.

"""Application-owned incident investigation data contracts — no fixture or proof semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class HypothesisId(StrEnum):
    """Scenario-domain hypothesis identifiers (H1/H2/H3)."""

    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


class TelemetryAvailability(StrEnum):
    """Typed telemetry source response — observation present vs absent for window."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class TelemetryUnavailabilityReason(StrEnum):
    """Bounded reason when telemetry source has no admissible observation."""

    NO_OBSERVATION_FOR_WINDOW = "no_observation_for_window"


class TimeWindowLabel(StrEnum):
    """Typed temporal windows for incident evidence (not a generic time-series engine)."""

    BASELINE = "baseline_week"
    BEFORE_INCIDENT = "before_incident"
    INCIDENT = "incident_window"
    COMPARISON = "comparison_high_load"


INCIDENT_WINDOW_START = datetime(2026, 3, 10, 6, 0, 0)
INCIDENT_WINDOW_END = datetime(2026, 3, 12, 18, 0, 0)
LINE_ID = "line4"
COMPARISON_LINE_ID = "line3"
STATION_ID = "complex_assembly_station"


@dataclass(frozen=True, slots=True)
class TimeWindow:
    label: str
    observed_from: datetime
    observed_to: datetime


@dataclass(frozen=True, slots=True)
class WorkloadObservation:
    line_id: str
    window: TimeWindow
    order_volume_delta_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class ThroughputObservation:
    line_id: str
    window: TimeWindow
    target_attainment_pct: float
    baseline_attainment_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class StaffingRecord:
    source_id: str
    line_id: str
    shift_id: str
    window: TimeWindow
    scheduled_headcount: int
    required_headcount: int
    record_generated_at: datetime
    record_valid_for: TimeWindow
    status: str


@dataclass(frozen=True, slots=True)
class AttendanceRecord:
    source_id: str
    line_id: str
    shift_id: str
    window: TimeWindow
    confirmed_headcount: int
    confirmed_at: datetime


@dataclass(frozen=True, slots=True)
class ComparisonObservation:
    comparison_line_id: str
    reference_line_id: str
    window: TimeWindow
    workload_delta_pct: float
    target_attainment_pct: float
    reference_attainment_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class TelemetryObservation:
    station_id: str
    window: TimeWindow
    availability: TelemetryAvailability
    signal_state: str | None = None
    complex_assembly_throughput_pct: float | None = None
    baseline_throughput_pct: float | None = None
    admissible: bool = True
    unavailability_reason: TelemetryUnavailabilityReason | None = None


@dataclass(frozen=True, slots=True)
class IncidentOperationalData:
    """Observable incident investigation data surface — application tool provider input."""

    workload_incident: WorkloadObservation
    workload_baseline: WorkloadObservation
    throughput_incident: ThroughputObservation
    throughput_before: ThroughputObservation
    throughput_baseline: ThroughputObservation
    staffing_preliminary: StaffingRecord
    staffing_attendance: AttendanceRecord
    comparison: ComparisonObservation
    telemetry: TelemetryObservation

    @property
    def station_id(self) -> str:
        return self.telemetry.station_id
