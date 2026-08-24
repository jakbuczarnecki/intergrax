# © Artur Czarnecki. All rights reserved.

"""Scenario production tools — registered on platform ToolRegistry; fixture behind provider."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from platform_proofs.scenarios.ai_incident_investigation.fixtures import (
    IncidentFixture,
    LINE_ID,
    STATION_ID,
    TelemetryAvailability,
    TimeWindowLabel,
)
from testing_support.builder import tools_agent_make_contract

TOOL_WORKLOAD_READ = "production.workload.read"
TOOL_THROUGHPUT_READ = "production.throughput.read"
TOOL_STAFFING_SCHEDULE_READ = "production.staffing.schedule.read"
TOOL_STAFFING_ATTENDANCE_READ = "production.staffing.attendance.read"
TOOL_COMPARISON_READ = "production.comparison.read"
TOOL_TELEMETRY_READ = "production.telemetry.read"

SCENARIO_TOOL_IDS: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_COMPARISON_READ,
    TOOL_TELEMETRY_READ,
)


class LineWindowInput(BaseModel):
    line_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class WorkloadOutput(BaseModel):
    line_id: str
    window: str
    order_volume_delta_pct: float
    observed_from: str
    observed_to: str
    admissible: bool


class ThroughputOutput(BaseModel):
    line_id: str
    window: str
    target_attainment_pct: float
    baseline_attainment_pct: float
    observed_from: str
    observed_to: str
    admissible: bool


class StaffingScheduleInput(BaseModel):
    line_id: str = Field(min_length=1)
    shift_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class StaffingScheduleOutput(BaseModel):
    source_id: str
    line_id: str
    shift_id: str
    window: str
    scheduled_headcount: int
    required_headcount: int
    record_generated_at: str
    record_valid_from: str
    record_valid_to: str
    window_observed_from: str
    window_observed_to: str
    status: str


class StaffingAttendanceInput(BaseModel):
    line_id: str = Field(min_length=1)
    shift_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class StaffingAttendanceOutput(BaseModel):
    source_id: str
    line_id: str
    shift_id: str
    window: str
    confirmed_headcount: int
    confirmed_at: str
    observed_from: str
    observed_to: str


class ComparisonInput(BaseModel):
    reference_line_id: str = Field(min_length=1)
    comparison_line_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class ComparisonOutput(BaseModel):
    comparison_line_id: str
    reference_line_id: str
    window: str
    workload_delta_pct: float
    comparison_attainment_pct: float
    reference_attainment_pct: float
    observed_from: str
    observed_to: str
    admissible: bool


class TelemetryInput(BaseModel):
    station_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class TelemetryOutput(BaseModel):
    station_id: str
    window: str
    availability: str
    signal_state: str | None = None
    complex_assembly_throughput_pct: float | None = None
    baseline_throughput_pct: float | None = None
    observed_from: str
    observed_to: str
    admissible: bool
    unavailability_reason: str | None = None


def _iso(dt: object) -> str:
    return dt.isoformat()


class _WorkloadHandler(ToolHandler[LineWindowInput, WorkloadOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(self, request: ToolExecutionRequest[LineWindowInput]) -> WorkloadOutput:
        if request.input.line_id != LINE_ID:
            return WorkloadOutput(
                line_id=request.input.line_id,
                window=request.input.window,
                order_volume_delta_pct=0.0,
                observed_from="",
                observed_to="",
                admissible=False,
            )
        if request.input.window == TimeWindowLabel.BASELINE:
            obs = self._fixture.workload_baseline
        elif request.input.window == TimeWindowLabel.INCIDENT:
            obs = self._fixture.workload_incident
        else:
            return WorkloadOutput(
                line_id=request.input.line_id,
                window=request.input.window,
                order_volume_delta_pct=0.0,
                observed_from="",
                observed_to="",
                admissible=False,
            )
        return WorkloadOutput(
            line_id=obs.line_id,
            window=obs.window.label,
            order_volume_delta_pct=obs.order_volume_delta_pct,
            observed_from=_iso(obs.window.observed_from),
            observed_to=_iso(obs.window.observed_to),
            admissible=obs.admissible,
        )


class _ThroughputHandler(ToolHandler[LineWindowInput, ThroughputOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(self, request: ToolExecutionRequest[LineWindowInput]) -> ThroughputOutput:
        if request.input.line_id != LINE_ID:
            return ThroughputOutput(
                line_id=request.input.line_id,
                window=request.input.window,
                target_attainment_pct=0.0,
                baseline_attainment_pct=0.0,
                observed_from="",
                observed_to="",
                admissible=False,
            )
        obs_map = {
            TimeWindowLabel.BASELINE: self._fixture.throughput_baseline,
            TimeWindowLabel.BEFORE_INCIDENT: self._fixture.throughput_before,
            TimeWindowLabel.INCIDENT: self._fixture.throughput_incident,
        }
        obs = obs_map.get(request.input.window)  # type: ignore[arg-type]
        if obs is None:
            return ThroughputOutput(
                line_id=request.input.line_id,
                window=request.input.window,
                target_attainment_pct=0.0,
                baseline_attainment_pct=0.0,
                observed_from="",
                observed_to="",
                admissible=False,
            )
        return ThroughputOutput(
            line_id=obs.line_id,
            window=obs.window.label,
            target_attainment_pct=obs.target_attainment_pct,
            baseline_attainment_pct=obs.baseline_attainment_pct,
            observed_from=_iso(obs.window.observed_from),
            observed_to=_iso(obs.window.observed_to),
            admissible=obs.admissible,
        )


class _StaffingScheduleHandler(ToolHandler[StaffingScheduleInput, StaffingScheduleOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(
        self, request: ToolExecutionRequest[StaffingScheduleInput]
    ) -> StaffingScheduleOutput:
        record = self._fixture.staffing_preliminary
        if (
            request.input.line_id != record.line_id
            or request.input.shift_id != record.shift_id
        ):
            return StaffingScheduleOutput(
                source_id="no_data",
                line_id=request.input.line_id,
                shift_id=request.input.shift_id,
                window=request.input.window,
                scheduled_headcount=0,
                required_headcount=0,
                record_generated_at="",
                record_valid_from="",
                record_valid_to="",
                window_observed_from="",
                window_observed_to="",
                status="unavailable",
            )
        return StaffingScheduleOutput(
            source_id=record.source_id,
            line_id=record.line_id,
            shift_id=record.shift_id,
            window=record.window.label,
            scheduled_headcount=record.scheduled_headcount,
            required_headcount=record.required_headcount,
            record_generated_at=_iso(record.record_generated_at),
            record_valid_from=_iso(record.record_valid_for.observed_from),
            record_valid_to=_iso(record.record_valid_for.observed_to),
            window_observed_from=_iso(record.window.observed_from),
            window_observed_to=_iso(record.window.observed_to),
            status=record.status,
        )


class _StaffingAttendanceHandler(ToolHandler[StaffingAttendanceInput, StaffingAttendanceOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(
        self, request: ToolExecutionRequest[StaffingAttendanceInput]
    ) -> StaffingAttendanceOutput:
        record = self._fixture.staffing_attendance
        if (
            request.input.line_id != record.line_id
            or request.input.shift_id != record.shift_id
        ):
            return StaffingAttendanceOutput(
                source_id="no_data",
                line_id=request.input.line_id,
                shift_id=request.input.shift_id,
                window=request.input.window,
                confirmed_headcount=0,
                confirmed_at="",
                observed_from="",
                observed_to="",
            )
        return StaffingAttendanceOutput(
            source_id=record.source_id,
            line_id=record.line_id,
            shift_id=record.shift_id,
            window=record.window.label,
            confirmed_headcount=record.confirmed_headcount,
            confirmed_at=_iso(record.confirmed_at),
            observed_from=_iso(record.window.observed_from),
            observed_to=_iso(record.window.observed_to),
        )


class _ComparisonHandler(ToolHandler[ComparisonInput, ComparisonOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(self, request: ToolExecutionRequest[ComparisonInput]) -> ComparisonOutput:
        obs = self._fixture.comparison
        if (
            request.input.reference_line_id != obs.reference_line_id
            or request.input.comparison_line_id != obs.comparison_line_id
        ):
            return ComparisonOutput(
                comparison_line_id=request.input.comparison_line_id,
                reference_line_id=request.input.reference_line_id,
                window=request.input.window,
                workload_delta_pct=0.0,
                comparison_attainment_pct=0.0,
                reference_attainment_pct=0.0,
                observed_from="",
                observed_to="",
                admissible=False,
            )
        return ComparisonOutput(
            comparison_line_id=obs.comparison_line_id,
            reference_line_id=obs.reference_line_id,
            window=obs.window.label,
            workload_delta_pct=obs.workload_delta_pct,
            comparison_attainment_pct=obs.target_attainment_pct,
            reference_attainment_pct=obs.reference_attainment_pct,
            observed_from=_iso(obs.window.observed_from),
            observed_to=_iso(obs.window.observed_to),
            admissible=obs.admissible,
        )


class _TelemetryHandler(ToolHandler[TelemetryInput, TelemetryOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(self, request: ToolExecutionRequest[TelemetryInput]) -> TelemetryOutput:
        obs = self._fixture.telemetry
        if request.input.station_id != obs.station_id:
            return TelemetryOutput(
                station_id=request.input.station_id,
                window=request.input.window,
                availability=TelemetryAvailability.UNAVAILABLE.value,
                observed_from="",
                observed_to="",
                admissible=False,
                unavailability_reason="no_observation_for_window",
            )
        if obs.availability is TelemetryAvailability.UNAVAILABLE:
            return TelemetryOutput(
                station_id=obs.station_id,
                window=obs.window.label,
                availability=TelemetryAvailability.UNAVAILABLE.value,
                observed_from=_iso(obs.window.observed_from),
                observed_to=_iso(obs.window.observed_to),
                admissible=obs.admissible,
                unavailability_reason=(
                    obs.unavailability_reason.value if obs.unavailability_reason else None
                ),
            )
        return TelemetryOutput(
            station_id=obs.station_id,
            window=obs.window.label,
            availability=TelemetryAvailability.AVAILABLE.value,
            signal_state=obs.signal_state,
            complex_assembly_throughput_pct=obs.complex_assembly_throughput_pct,
            baseline_throughput_pct=obs.baseline_throughput_pct,
            observed_from=_iso(obs.window.observed_from),
            observed_to=_iso(obs.window.observed_to),
            admissible=obs.admissible,
        )


def register_scenario_tools(registry: ToolRegistry, fixture: IncidentFixture) -> None:
    registry.register(
        tools_agent_make_contract(TOOL_WORKLOAD_READ, LineWindowInput, WorkloadOutput),
        _WorkloadHandler(fixture),
    )
    registry.register(
        tools_agent_make_contract(TOOL_THROUGHPUT_READ, LineWindowInput, ThroughputOutput),
        _ThroughputHandler(fixture),
    )
    registry.register(
        tools_agent_make_contract(
            TOOL_STAFFING_SCHEDULE_READ, StaffingScheduleInput, StaffingScheduleOutput
        ),
        _StaffingScheduleHandler(fixture),
    )
    registry.register(
        tools_agent_make_contract(
            TOOL_STAFFING_ATTENDANCE_READ, StaffingAttendanceInput, StaffingAttendanceOutput
        ),
        _StaffingAttendanceHandler(fixture),
    )
    registry.register(
        tools_agent_make_contract(TOOL_COMPARISON_READ, ComparisonInput, ComparisonOutput),
        _ComparisonHandler(fixture),
    )
    registry.register(
        tools_agent_make_contract(TOOL_TELEMETRY_READ, TelemetryInput, TelemetryOutput),
        _TelemetryHandler(fixture),
    )


def default_line_window_input() -> dict[str, str]:
    return {"line_id": LINE_ID, "window": TimeWindowLabel.INCIDENT}


def default_staffing_input(shift_id: str = "shift_b") -> dict[str, str]:
    return {
        "line_id": LINE_ID,
        "shift_id": shift_id,
        "window": TimeWindowLabel.INCIDENT,
    }


def default_comparison_input() -> dict[str, str]:
    return {
        "reference_line_id": LINE_ID,
        "comparison_line_id": "line3",
        "window": TimeWindowLabel.COMPARISON,
    }


def default_telemetry_input(station_id: str = STATION_ID) -> dict[str, str]:
    return {"station_id": station_id, "window": TimeWindowLabel.INCIDENT}
