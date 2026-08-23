# © Artur Czarnecki. All rights reserved.

"""Synthetic manufacturing fixture — observable surface vs private evaluator truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class HypothesisId(StrEnum):
    """Scenario-local hypothesis identifiers (H1/H2/H3)."""

    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


class TimeWindowLabel(StrEnum):
    """Typed temporal windows for scenario evidence (not a generic time-series engine)."""

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
class _PrivateFixtureTruth:
    """Evaluator-only ground truth — never model-visible."""

    initiating_factor_code: str
    station_id: str
    expected_hypothesis: HypothesisId


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
    signal_state: str
    complex_assembly_throughput_pct: float
    baseline_throughput_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class IncidentFixture:
    """Scenario fixture with strict observable / hidden separation."""

    private_truth: _PrivateFixtureTruth
    workload_incident: WorkloadObservation
    workload_baseline: WorkloadObservation
    throughput_incident: ThroughputObservation
    throughput_before: ThroughputObservation
    throughput_baseline: ThroughputObservation
    staffing_preliminary: StaffingRecord
    staffing_attendance: AttendanceRecord
    comparison: ComparisonObservation
    telemetry: TelemetryObservation


def _baseline_window() -> TimeWindow:
    return TimeWindow(
        label=TimeWindowLabel.BASELINE,
        observed_from=datetime(2026, 3, 2, 6, 0, 0),
        observed_to=datetime(2026, 3, 6, 18, 0, 0),
    )


def _before_incident_window() -> TimeWindow:
    return TimeWindow(
        label=TimeWindowLabel.BEFORE_INCIDENT,
        observed_from=datetime(2026, 3, 9, 6, 0, 0),
        observed_to=datetime(2026, 3, 9, 18, 0, 0),
    )


def _incident_window() -> TimeWindow:
    return TimeWindow(
        label=TimeWindowLabel.INCIDENT,
        observed_from=INCIDENT_WINDOW_START,
        observed_to=INCIDENT_WINDOW_END,
    )


def _comparison_window() -> TimeWindow:
    return TimeWindow(
        label=TimeWindowLabel.COMPARISON,
        observed_from=datetime(2026, 2, 24, 6, 0, 0),
        observed_to=datetime(2026, 2, 26, 18, 0, 0),
    )


def _prior_week_roster_window() -> TimeWindow:
    return TimeWindow(
        label="prior_week_roster",
        observed_from=datetime(2026, 3, 3, 6, 0, 0),
        observed_to=datetime(2026, 3, 7, 18, 0, 0),
    )


def build_resolved_fixture() -> IncidentFixture:
    """Full RESOLVED evidence world: H1/H2/H3 adversarial dataset with temporal structure."""
    incident = _incident_window()
    private_truth = _PrivateFixtureTruth(
        initiating_factor_code="station_signal_degraded_pattern",
        station_id=STATION_ID,
        expected_hypothesis=HypothesisId.H3,
    )
    return IncidentFixture(
        private_truth=private_truth,
        workload_incident=WorkloadObservation(
            line_id=LINE_ID,
            window=incident,
            order_volume_delta_pct=22.0,
            admissible=True,
        ),
        workload_baseline=WorkloadObservation(
            line_id=LINE_ID,
            window=_baseline_window(),
            order_volume_delta_pct=1.5,
            admissible=True,
        ),
        throughput_incident=ThroughputObservation(
            line_id=LINE_ID,
            window=incident,
            target_attainment_pct=78.0,
            baseline_attainment_pct=94.0,
            admissible=True,
        ),
        throughput_before=ThroughputObservation(
            line_id=LINE_ID,
            window=_before_incident_window(),
            target_attainment_pct=86.0,
            baseline_attainment_pct=94.0,
            admissible=True,
        ),
        throughput_baseline=ThroughputObservation(
            line_id=LINE_ID,
            window=_baseline_window(),
            target_attainment_pct=94.0,
            baseline_attainment_pct=94.0,
            admissible=True,
        ),
        staffing_preliminary=StaffingRecord(
            source_id="shift_roster_planning_export",
            line_id=LINE_ID,
            shift_id="shift_b",
            window=incident,
            scheduled_headcount=8,
            required_headcount=12,
            record_generated_at=datetime(2026, 3, 3, 8, 0, 0),
            record_valid_for=_prior_week_roster_window(),
            status="preliminary_export",
        ),
        staffing_attendance=AttendanceRecord(
            source_id="time_attendance_confirmed",
            line_id=LINE_ID,
            shift_id="shift_b",
            window=incident,
            confirmed_headcount=12,
            confirmed_at=datetime(2026, 3, 11, 7, 30, 0),
        ),
        comparison=ComparisonObservation(
            comparison_line_id=COMPARISON_LINE_ID,
            reference_line_id=LINE_ID,
            window=_comparison_window(),
            workload_delta_pct=24.0,
            target_attainment_pct=93.0,
            reference_attainment_pct=78.0,
            admissible=True,
        ),
        telemetry=TelemetryObservation(
            station_id=STATION_ID,
            window=incident,
            signal_state="intermittent_degraded",
            complex_assembly_throughput_pct=62.0,
            baseline_throughput_pct=91.0,
            admissible=True,
        ),
    )


def build_skeleton_fixture() -> IncidentFixture:
    """Backward-compatible alias — full RESOLVED fixture replaces minimal skeleton slice."""
    return build_resolved_fixture()


def staffing_record_admissible_for_incident(record: StaffingRecord) -> bool:
    """Scenario-local admissibility: preliminary roster valid window must overlap incident."""
    incident = _incident_window()
    return (
        record.record_valid_for.observed_from <= incident.observed_to
        and record.record_valid_for.observed_to >= incident.observed_from
    )


# Strings that must never appear in model-visible/tool-visible material.
FORBIDDEN_LEAK_MARKERS: frozenset[str] = frozenset(
    {
        "station_signal_degraded_pattern",
        "initiating_factor_code",
        "hidden_root_cause",
        "correct_answer",
        "expected_diagnosis",
        "expected_hypothesis",
        "fixture_hidden_truth",
        "_PrivateFixtureTruth",
    }
)
