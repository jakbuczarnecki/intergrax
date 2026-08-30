# © Artur Czarnecki. All rights reserved.

"""Scenario production tools — registered on platform ToolRegistry; fixture behind provider."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import (
    attendance_meets_required,
    comparison_weakens_overload,
    h1_initially_plausible,
    is_high_workload,
    is_material_throughput_drop,
    observations_from_evidence_nodes,
    parse_comparison_payload,
    parse_staffing_attendance_payload,
    parse_staffing_schedule_payload,
    parse_telemetry_payload,
    parse_throughput_payload,
    parse_workload_payload,
    preliminary_suggests_shortage,
    staffing_record_admissible_for_incident,
    staffing_shortage_confirmed,
    telemetry_is_unavailable,
    telemetry_supports_degradation,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    COMPARISON_LINE_ID,
    IncidentOperationalData,
    LINE_ID,
    STATION_ID,
    TelemetryAvailability,
    TimeWindowLabel,
)
TOOL_WORKLOAD_READ = "production.workload.read"
TOOL_THROUGHPUT_READ = "production.throughput.read"
TOOL_STAFFING_SCHEDULE_READ = "production.staffing.schedule.read"
TOOL_STAFFING_ATTENDANCE_READ = "production.staffing.attendance.read"
TOOL_COMPARISON_READ = "production.comparison.read"
TOOL_TELEMETRY_READ = "production.telemetry.read"

TOOL_WORKLOAD_EVALUATE = "incident.workload.evaluate"
TOOL_STAFFING_EVALUATE = "incident.staffing.evaluate"
TOOL_COMPARISON_EVALUATE = "incident.comparison.evaluate"
TOOL_TELEMETRY_EVALUATE = "incident.telemetry.evaluate"

RAW_EVIDENCE_TOOL_IDS: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_COMPARISON_READ,
    TOOL_TELEMETRY_READ,
)

ANALYSIS_TOOL_IDS: tuple[str, ...] = (
    TOOL_WORKLOAD_EVALUATE,
    TOOL_STAFFING_EVALUATE,
    TOOL_COMPARISON_EVALUATE,
    TOOL_TELEMETRY_EVALUATE,
)

SCENARIO_TOOL_IDS: tuple[str, ...] = RAW_EVIDENCE_TOOL_IDS + ANALYSIS_TOOL_IDS


class ScenarioEvidenceStore:
    """Mutable scenario evidence context populated by canonical tool reads."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, object]] = {}

    def record(
        self,
        evidence_id: str,
        payload: dict[str, object],
        *,
        source_tool_id: str,
    ) -> None:
        self._nodes[evidence_id] = {
            "evidence_id": evidence_id,
            "payload": payload,
            "source_tool_id": source_tool_id,
        }

    def evidence_nodes(self) -> tuple[dict[str, object], ...]:
        return tuple(self._nodes.values())

    def get_payload(self, evidence_id: str) -> dict[str, object] | None:
        node = self._nodes.get(evidence_id)
        if node is None:
            return None
        payload = node.get("payload")
        return dict(payload) if isinstance(payload, dict) else None


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


class EvidenceRefInput(BaseModel):
    evidence_id: str = Field(min_length=1)


class WorkloadEvaluateInput(BaseModel):
    workload_evidence_id: str = Field(min_length=1)
    throughput_evidence_id: str = Field(min_length=1)


class WorkloadEvaluateOutput(BaseModel):
    analysis_type: Literal["workload_throughput_correlation"] = "workload_throughput_correlation"
    source_evidence_ids: tuple[str, ...]
    workload_pressure_observed: bool | None
    throughput_degradation_observed: bool | None
    correlation_present: bool | None
    causal_conclusion_allowed: Literal[False] = False


class StaffingEvaluateInput(BaseModel):
    schedule_evidence_id: str = Field(min_length=1)
    attendance_evidence_id: str | None = None


class StaffingEvaluateOutput(BaseModel):
    analysis_type: Literal["staffing_consistency"] = "staffing_consistency"
    source_evidence_ids: tuple[str, ...]
    schedule_admissible: bool | None
    preliminary_shortage_signal: bool | None
    attendance_available: bool
    attendance_meets_required: bool | None
    shortage_confirmed: bool | None
    conflict_detected: bool | None


class ComparisonEvaluateInput(BaseModel):
    workload_evidence_id: str = Field(min_length=1)
    throughput_evidence_id: str = Field(min_length=1)
    comparison_evidence_id: str = Field(min_length=1)


class ComparisonEvaluateOutput(BaseModel):
    analysis_type: Literal["peer_line_comparison"] = "peer_line_comparison"
    source_evidence_ids: tuple[str, ...]
    comparison_available: bool
    peer_workload_similar: bool | None
    peer_throughput_stable: bool | None
    overload_only_explanation_weakened: bool | None


class TelemetryEvaluateInput(BaseModel):
    telemetry_evidence_id: str = Field(min_length=1)


class TelemetryEvaluateOutput(BaseModel):
    analysis_type: Literal["station_telemetry"] = "station_telemetry"
    source_evidence_ids: tuple[str, ...]
    telemetry_available: bool
    telemetry_unavailable: bool | None
    degradation_detected: bool | None
    intermittent_pattern: bool | None


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


def _analysis_evidence_id(analysis_type: str, source_ids: tuple[str, ...]) -> str:
    suffix = "_".join(source_id.replace(".", "_") for source_id in source_ids)
    return f"evidence.analysis.{analysis_type}.{suffix}"


class _WorkloadEvaluateHandler(ToolHandler[WorkloadEvaluateInput, WorkloadEvaluateOutput]):
    def __init__(self, evidence_store: ScenarioEvidenceStore) -> None:
        self._evidence_store = evidence_store

    def execute(self, request: ToolExecutionRequest[WorkloadEvaluateInput]) -> WorkloadEvaluateOutput:
        source_ids = (
            request.input.workload_evidence_id,
            request.input.throughput_evidence_id,
        )
        workload_payload = self._evidence_store.get_payload(request.input.workload_evidence_id)
        throughput_payload = self._evidence_store.get_payload(request.input.throughput_evidence_id)
        if workload_payload is None or throughput_payload is None:
            return WorkloadEvaluateOutput(
                source_evidence_ids=source_ids,
                workload_pressure_observed=None,
                throughput_degradation_observed=None,
                correlation_present=None,
            )
        workload = parse_workload_payload(workload_payload)
        throughput = parse_throughput_payload(throughput_payload)
        pressure = is_high_workload(workload)
        degradation = is_material_throughput_drop(throughput)
        correlation = h1_initially_plausible(workload, throughput)
        return WorkloadEvaluateOutput(
            source_evidence_ids=source_ids,
            workload_pressure_observed=pressure,
            throughput_degradation_observed=degradation,
            correlation_present=correlation,
        )


class _StaffingEvaluateHandler(ToolHandler[StaffingEvaluateInput, StaffingEvaluateOutput]):
    def __init__(self, evidence_store: ScenarioEvidenceStore) -> None:
        self._evidence_store = evidence_store

    def execute(self, request: ToolExecutionRequest[StaffingEvaluateInput]) -> StaffingEvaluateOutput:
        source_ids: list[str] = [request.input.schedule_evidence_id]
        schedule_payload = self._evidence_store.get_payload(request.input.schedule_evidence_id)
        if schedule_payload is None:
            return StaffingEvaluateOutput(
                source_evidence_ids=tuple(source_ids),
                schedule_admissible=None,
                preliminary_shortage_signal=None,
                attendance_available=False,
                attendance_meets_required=None,
                shortage_confirmed=None,
                conflict_detected=None,
            )
        schedule = parse_staffing_schedule_payload(schedule_payload)
        schedule_admissible = staffing_record_admissible_for_incident(schedule)
        preliminary_shortage = preliminary_suggests_shortage(schedule)
        attendance_payload = None
        if request.input.attendance_evidence_id:
            source_ids.append(request.input.attendance_evidence_id)
            attendance_payload = self._evidence_store.get_payload(
                request.input.attendance_evidence_id
            )
        if attendance_payload is None:
            return StaffingEvaluateOutput(
                source_evidence_ids=tuple(source_ids),
                schedule_admissible=schedule_admissible,
                preliminary_shortage_signal=preliminary_shortage,
                attendance_available=False,
                attendance_meets_required=None,
                shortage_confirmed=None,
                conflict_detected=None,
            )
        attendance = parse_staffing_attendance_payload(attendance_payload)
        meets_required = attendance_meets_required(schedule, attendance)
        shortage = staffing_shortage_confirmed(schedule, attendance)
        conflict = preliminary_shortage and meets_required
        return StaffingEvaluateOutput(
            source_evidence_ids=tuple(source_ids),
            schedule_admissible=schedule_admissible,
            preliminary_shortage_signal=preliminary_shortage,
            attendance_available=True,
            attendance_meets_required=meets_required,
            shortage_confirmed=shortage,
            conflict_detected=conflict,
        )


class _ComparisonEvaluateHandler(ToolHandler[ComparisonEvaluateInput, ComparisonEvaluateOutput]):
    def __init__(self, evidence_store: ScenarioEvidenceStore) -> None:
        self._evidence_store = evidence_store

    def execute(
        self, request: ToolExecutionRequest[ComparisonEvaluateInput]
    ) -> ComparisonEvaluateOutput:
        source_ids = (
            request.input.workload_evidence_id,
            request.input.throughput_evidence_id,
            request.input.comparison_evidence_id,
        )
        workload_payload = self._evidence_store.get_payload(request.input.workload_evidence_id)
        throughput_payload = self._evidence_store.get_payload(request.input.throughput_evidence_id)
        comparison_payload = self._evidence_store.get_payload(request.input.comparison_evidence_id)
        if (
            workload_payload is None
            or throughput_payload is None
            or comparison_payload is None
        ):
            return ComparisonEvaluateOutput(
                source_evidence_ids=source_ids,
                comparison_available=False,
                peer_workload_similar=None,
                peer_throughput_stable=None,
                overload_only_explanation_weakened=None,
            )
        workload = parse_workload_payload(workload_payload)
        throughput = parse_throughput_payload(throughput_payload)
        comparison = parse_comparison_payload(comparison_payload)
        weakened = comparison_weakens_overload(workload, throughput, comparison)
        return ComparisonEvaluateOutput(
            source_evidence_ids=source_ids,
            comparison_available=comparison.admissible,
            peer_workload_similar=comparison.admissible,
            peer_throughput_stable=comparison.admissible,
            overload_only_explanation_weakened=weakened,
        )


class _TelemetryEvaluateHandler(ToolHandler[TelemetryEvaluateInput, TelemetryEvaluateOutput]):
    def __init__(self, evidence_store: ScenarioEvidenceStore) -> None:
        self._evidence_store = evidence_store

    def execute(self, request: ToolExecutionRequest[TelemetryEvaluateInput]) -> TelemetryEvaluateOutput:
        source_ids = (request.input.telemetry_evidence_id,)
        telemetry_payload = self._evidence_store.get_payload(request.input.telemetry_evidence_id)
        if telemetry_payload is None:
            return TelemetryEvaluateOutput(
                source_evidence_ids=source_ids,
                telemetry_available=False,
                telemetry_unavailable=None,
                degradation_detected=None,
                intermittent_pattern=None,
            )
        telemetry = parse_telemetry_payload(telemetry_payload)
        unavailable = telemetry_is_unavailable(telemetry)
        if unavailable:
            return TelemetryEvaluateOutput(
                source_evidence_ids=source_ids,
                telemetry_available=False,
                telemetry_unavailable=True,
                degradation_detected=None,
                intermittent_pattern=None,
            )
        degraded = telemetry_supports_degradation(telemetry)
        intermittent = telemetry.signal_state in {
            "intermittent_degraded",
            "intermittent_fault",
        }
        return TelemetryEvaluateOutput(
            source_evidence_ids=source_ids,
            telemetry_available=True,
            telemetry_unavailable=False,
            degradation_detected=degraded,
            intermittent_pattern=intermittent,
        )


def _scenario_tool_contract(
    tool_id: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    *,
    description: str,
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=description,
        input_schema=input_model,
        output_schema=output_model,
        error_mapping={},
        side_effects=False,
    )


class _WorkloadHandler(ToolHandler[LineWindowInput, WorkloadOutput]):
    def __init__(self, operational_data: IncidentOperationalData) -> None:
        self._operational_data = operational_data

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
            obs = self._operational_data.workload_baseline
        elif request.input.window == TimeWindowLabel.INCIDENT:
            obs = self._operational_data.workload_incident
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
    def __init__(self, operational_data: IncidentOperationalData) -> None:
        self._operational_data = operational_data

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
            TimeWindowLabel.BASELINE: self._operational_data.throughput_baseline,
            TimeWindowLabel.BEFORE_INCIDENT: self._operational_data.throughput_before,
            TimeWindowLabel.INCIDENT: self._operational_data.throughput_incident,
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
    def __init__(self, operational_data: IncidentOperationalData) -> None:
        self._operational_data = operational_data

    def execute(
        self, request: ToolExecutionRequest[StaffingScheduleInput]
    ) -> StaffingScheduleOutput:
        record = self._operational_data.staffing_preliminary
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
    def __init__(self, operational_data: IncidentOperationalData) -> None:
        self._operational_data = operational_data

    def execute(
        self, request: ToolExecutionRequest[StaffingAttendanceInput]
    ) -> StaffingAttendanceOutput:
        record = self._operational_data.staffing_attendance
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
    def __init__(self, operational_data: IncidentOperationalData) -> None:
        self._operational_data = operational_data

    def execute(self, request: ToolExecutionRequest[ComparisonInput]) -> ComparisonOutput:
        obs = self._operational_data.comparison
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
    def __init__(self, operational_data: IncidentOperationalData) -> None:
        self._operational_data = operational_data

    def execute(self, request: ToolExecutionRequest[TelemetryInput]) -> TelemetryOutput:
        obs = self._operational_data.telemetry
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


def register_scenario_tools(
    registry: ToolRegistry,
    operational_data: IncidentOperationalData,
    *,
    evidence_store: ScenarioEvidenceStore | None = None,
) -> ScenarioEvidenceStore:
    store = evidence_store or ScenarioEvidenceStore()
    registry.register(
        _scenario_tool_contract(
            TOOL_WORKLOAD_READ,
            LineWindowInput,
            WorkloadOutput,
            description="Read production workload observations for a line and time window.",
        ),
        _WorkloadHandler(operational_data),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_WORKLOAD_EVALUATE,
            WorkloadEvaluateInput,
            WorkloadEvaluateOutput,
            description=(
                "Evaluate whether workload pressure and throughput degradation are "
                "observable from referenced incident evidence IDs."
            ),
        ),
        _WorkloadEvaluateHandler(store),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_THROUGHPUT_READ,
            LineWindowInput,
            ThroughputOutput,
            description="Read production throughput observations for a line and time window.",
        ),
        _ThroughputHandler(operational_data),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_STAFFING_SCHEDULE_READ,
            StaffingScheduleInput,
            StaffingScheduleOutput,
            description="Read preliminary staffing schedule export for a line shift window.",
        ),
        _StaffingScheduleHandler(operational_data),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_STAFFING_ATTENDANCE_READ,
            StaffingAttendanceInput,
            StaffingAttendanceOutput,
            description="Read confirmed staffing attendance for a line shift window.",
        ),
        _StaffingAttendanceHandler(operational_data),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_STAFFING_EVALUATE,
            StaffingEvaluateInput,
            StaffingEvaluateOutput,
            description=(
                "Evaluate whether staffing attendance confirms, contradicts, or leaves "
                "unresolved a preliminary staffing shortage signal using referenced evidence."
            ),
        ),
        _StaffingEvaluateHandler(store),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_COMPARISON_READ,
            ComparisonInput,
            ComparisonOutput,
            description=(
                "Read cross-line workload and attainment comparison observations. "
                "reference_line_id is the incident line under investigation; "
                "comparison_line_id is the permitted peer line to contrast against it; "
                "window must be comparison_high_load."
            ),
        ),
        _ComparisonHandler(operational_data),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_COMPARISON_EVALUATE,
            ComparisonEvaluateInput,
            ComparisonEvaluateOutput,
            description=(
                "Evaluate whether peer-line comparison weakens an overload-only explanation "
                "using referenced workload, throughput, and comparison evidence."
            ),
        ),
        _ComparisonEvaluateHandler(store),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_TELEMETRY_READ,
            TelemetryInput,
            TelemetryOutput,
            description="Read station telemetry observations for a time window.",
        ),
        _TelemetryHandler(operational_data),
    )
    registry.register(
        _scenario_tool_contract(
            TOOL_TELEMETRY_EVALUATE,
            TelemetryEvaluateInput,
            TelemetryEvaluateOutput,
            description=(
                "Evaluate whether station telemetry shows degradation, intermittent patterns, "
                "or unavailability for the referenced telemetry evidence ID."
            ),
        ),
        _TelemetryEvaluateHandler(store),
    )
    return store


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
