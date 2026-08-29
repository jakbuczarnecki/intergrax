# © Artur Czarnecki. All rights reserved.

"""APP-2BC-R1 deterministic domain analysis tool tests."""

from __future__ import annotations

import inspect

import pytest

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import build_resolved_fixture
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    TOOL_COMPARISON_EVALUATE,
    TOOL_STAFFING_EVALUATE,
    TOOL_TELEMETRY_EVALUATE,
    TOOL_WORKLOAD_EVALUATE,
    ComparisonEvaluateInput,
    LineWindowInput,
    StaffingAttendanceInput,
    StaffingEvaluateInput,
    StaffingScheduleInput,
    ComparisonInput,
    TelemetryEvaluateInput,
    TelemetryInput,
    WorkloadEvaluateInput,
    register_scenario_tools,
    default_comparison_input,
    default_line_window_input,
    default_staffing_input,
    default_telemetry_input,
)

pytestmark = pytest.mark.unit


def _request(tool_id: str, input_model):
    return ToolExecutionRequest(
        run_id="run-test",
        step_id="step-test",
        tool_id=tool_id,
        input=input_model,
    )


def _store_with_resolved_fixture():
    fixture = build_resolved_fixture()
    registry = ToolRegistry()
    store = register_scenario_tools(registry, fixture.to_operational_data())
    line_input = LineWindowInput.model_validate(default_line_window_input())
    staffing_input = StaffingScheduleInput.model_validate(default_staffing_input())
    attendance_input = StaffingAttendanceInput.model_validate(default_staffing_input())
    comparison_input = ComparisonInput.model_validate(default_comparison_input())
    telemetry_input = TelemetryInput.model_validate(default_telemetry_input())

    workload = registry.get("production.workload.read").handler.execute(
        _request("production.workload.read", line_input)
    )
    throughput = registry.get("production.throughput.read").handler.execute(
        _request("production.throughput.read", line_input)
    )
    schedule = registry.get("production.staffing.schedule.read").handler.execute(
        _request("production.staffing.schedule.read", staffing_input)
    )
    attendance = registry.get("production.staffing.attendance.read").handler.execute(
        _request("production.staffing.attendance.read", attendance_input)
    )
    comparison = registry.get("production.comparison.read").handler.execute(
        _request("production.comparison.read", comparison_input)
    )
    telemetry = registry.get("production.telemetry.read").handler.execute(
        _request("production.telemetry.read", telemetry_input)
    )
    store.record(str(WORKLOAD_EVIDENCE_ID), workload.model_dump(), source_tool_id="production.workload.read")
    store.record(
        str(THROUGHPUT_EVIDENCE_ID),
        throughput.model_dump(),
        source_tool_id="production.throughput.read",
    )
    store.record(
        str(STAFFING_PRELIMINARY_EVIDENCE_ID),
        schedule.model_dump(),
        source_tool_id="production.staffing.schedule.read",
    )
    store.record(
        str(STAFFING_ATTENDANCE_EVIDENCE_ID),
        attendance.model_dump(),
        source_tool_id="production.staffing.attendance.read",
    )
    store.record(
        str(COMPARISON_EVIDENCE_ID),
        comparison.model_dump(),
        source_tool_id="production.comparison.read",
    )
    store.record(
        str(TELEMETRY_EVIDENCE_ID),
        telemetry.model_dump(),
        source_tool_id="production.telemetry.read",
    )
    return registry, store


def test_workload_analysis_derived_from_evidence() -> None:
    registry, _store = _store_with_resolved_fixture()
    handler = registry.get(TOOL_WORKLOAD_EVALUATE).handler
    result = handler.execute(
        _request(
            TOOL_WORKLOAD_EVALUATE,
            WorkloadEvaluateInput(
                workload_evidence_id=str(WORKLOAD_EVIDENCE_ID),
                throughput_evidence_id=str(THROUGHPUT_EVIDENCE_ID),
            ),
        )
    )
    assert result.correlation_present is True
    assert result.causal_conclusion_allowed is False
    assert str(WORKLOAD_EVIDENCE_ID) in result.source_evidence_ids


def test_staffing_analysis_schedule_only() -> None:
    registry, store = _store_with_resolved_fixture()
    handler = registry.get(TOOL_STAFFING_EVALUATE).handler
    result = handler.execute(
        _request(
            TOOL_STAFFING_EVALUATE,
            StaffingEvaluateInput(schedule_evidence_id=str(STAFFING_PRELIMINARY_EVIDENCE_ID)),
        )
    )
    assert result.attendance_available is False
    assert result.shortage_confirmed is None


def test_staffing_analysis_with_attendance() -> None:
    registry, _store = _store_with_resolved_fixture()
    handler = registry.get(TOOL_STAFFING_EVALUATE).handler
    result = handler.execute(
        _request(
            TOOL_STAFFING_EVALUATE,
            StaffingEvaluateInput(
                schedule_evidence_id=str(STAFFING_PRELIMINARY_EVIDENCE_ID),
                attendance_evidence_id=str(STAFFING_ATTENDANCE_EVIDENCE_ID),
            ),
        )
    )
    assert result.attendance_available is True
    assert result.attendance_meets_required is True
    assert result.shortage_confirmed is False


def test_comparison_analysis_weakens_overload_only() -> None:
    registry, _store = _store_with_resolved_fixture()
    handler = registry.get(TOOL_COMPARISON_EVALUATE).handler
    result = handler.execute(
        _request(
            TOOL_COMPARISON_EVALUATE,
            ComparisonEvaluateInput(
                workload_evidence_id=str(WORKLOAD_EVIDENCE_ID),
                throughput_evidence_id=str(THROUGHPUT_EVIDENCE_ID),
                comparison_evidence_id=str(COMPARISON_EVIDENCE_ID),
            ),
        )
    )
    assert result.overload_only_explanation_weakened is True


def test_telemetry_analysis_detects_degradation() -> None:
    registry, _store = _store_with_resolved_fixture()
    handler = registry.get(TOOL_TELEMETRY_EVALUATE).handler
    result = handler.execute(
        _request(
            TOOL_TELEMETRY_EVALUATE,
            TelemetryEvaluateInput(telemetry_evidence_id=str(TELEMETRY_EVIDENCE_ID)),
        )
    )
    assert result.degradation_detected is True


def test_analysis_tools_do_not_emit_diagnosis_fields() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
        ComparisonEvaluateOutput,
        StaffingEvaluateOutput,
        TelemetryEvaluateOutput,
        WorkloadEvaluateOutput,
    )

    for model in (
        WorkloadEvaluateOutput,
        StaffingEvaluateOutput,
        ComparisonEvaluateOutput,
        TelemetryEvaluateOutput,
    ):
        fields = set(model.model_fields)
        assert "preferred_hypothesis" not in fields
        assert "root_cause" not in fields
        assert "resolution" not in fields


def test_analysis_tool_count_bounded() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.application.tools import ANALYSIS_TOOL_IDS

    assert 3 <= len(ANALYSIS_TOOL_IDS) <= 5
