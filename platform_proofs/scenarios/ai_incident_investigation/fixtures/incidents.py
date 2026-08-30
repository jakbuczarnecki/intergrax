# © Artur Czarnecki. All rights reserved.

"""Synthetic manufacturing fixture — observable surface vs private evaluator truth."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    AttendanceRecord,
    COMPARISON_LINE_ID,
    ComparisonObservation,
    HypothesisId,
    IncidentOperationalData,
    LINE_ID,
    STATION_ID,
    StaffingRecord,
    TelemetryAvailability,
    TelemetryObservation,
    TelemetryUnavailabilityReason,
    ThroughputObservation,
    TimeWindow,
    TimeWindowLabel,
    WorkloadObservation,
)


class ScenarioVariant(StrEnum):
    """Canonical scenario execution paths sharing one incident world."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class _PrivateFixtureTruth:
    """Evaluator-only ground truth — never model-visible."""

    initiating_factor_code: str
    station_id: str
    expected_hypothesis: HypothesisId


@dataclass(frozen=True, slots=True)
class IncidentFixture:
    """Scenario fixture with strict observable / hidden separation."""

    variant: ScenarioVariant
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

    def to_operational_data(self) -> IncidentOperationalData:
        return IncidentOperationalData(
            workload_incident=self.workload_incident,
            workload_baseline=self.workload_baseline,
            throughput_incident=self.throughput_incident,
            throughput_before=self.throughput_before,
            throughput_baseline=self.throughput_baseline,
            staffing_preliminary=self.staffing_preliminary,
            staffing_attendance=self.staffing_attendance,
            comparison=self.comparison,
            telemetry=self.telemetry,
        )


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
        observed_from=datetime(2026, 3, 10, 6, 0, 0),
        observed_to=datetime(2026, 3, 12, 18, 0, 0),
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
        variant=ScenarioVariant.RESOLVED,
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
            availability=TelemetryAvailability.AVAILABLE,
            signal_state="intermittent_degraded",
            complex_assembly_throughput_pct=62.0,
            baseline_throughput_pct=91.0,
            admissible=True,
        ),
    )


def build_unresolved_fixture() -> IncidentFixture:
    """FULL-2 evidence world: same incident, decisive telemetry unavailable for window."""
    resolved = build_resolved_fixture()
    incident = _incident_window()
    return IncidentFixture(
        variant=ScenarioVariant.UNRESOLVED,
        private_truth=resolved.private_truth,
        workload_incident=resolved.workload_incident,
        workload_baseline=resolved.workload_baseline,
        throughput_incident=resolved.throughput_incident,
        throughput_before=resolved.throughput_before,
        throughput_baseline=resolved.throughput_baseline,
        staffing_preliminary=resolved.staffing_preliminary,
        staffing_attendance=resolved.staffing_attendance,
        comparison=resolved.comparison,
        telemetry=TelemetryObservation(
            station_id=STATION_ID,
            window=incident,
            availability=TelemetryAvailability.UNAVAILABLE,
            unavailability_reason=TelemetryUnavailabilityReason.NO_OBSERVATION_FOR_WINDOW,
            admissible=True,
        ),
    )


def build_skeleton_fixture() -> IncidentFixture:
    """Backward-compatible alias — full RESOLVED fixture replaces minimal skeleton slice."""
    return build_resolved_fixture()


def staffing_record_admissible_for_incident(record: StaffingRecord) -> bool:
    """Fixture-local admissibility: preliminary roster valid window must overlap incident."""
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
