# © Artur Czarnecki. All rights reserved.

"""AI Incident completion eligibility gate tests (DS-E2E-15B.2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.decision_system.completion_eligibility import CompletionEligibilityStatus
from intergrax.decision_system.evidence_requirements import (
    EvidenceRequirementProviderAbsentError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_completion_gate import (
    CompletionEligibilityBlockedError,
    CompletionEligibilityGateConfig,
    assert_ai_incident_completion_eligible,
    evaluate_ai_incident_completion_eligibility,
)
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_requirement_semantics import (
    AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID,
    AI_INCIDENT_TELEMETRY_EVIDENCE_REQUIREMENT_ID,
    RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    STAFFING_ATTENDANCE_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import (
    evaluate_scenario_run,
)

pytestmark = pytest.mark.unit


def _nodes_without_attendance(nodes: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    attendance_id = str(STAFFING_ATTENDANCE_EVIDENCE_ID)
    return tuple(
        node for node in nodes if str(node.get("evidence_id")) != attendance_id
    )


@pytest.mark.asyncio
async def test_ai_incident_success_control_remains_eligible() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    result = await execute_resolved_skeleton(fixture_bundle.bundle)
    decision = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=result.evidence_nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    assert decision.status is CompletionEligibilityStatus.ELIGIBLE
    evaluation = evaluate_scenario_run(
        replace(result, tool_trace_count=1),
        fixture_bundle.fixture,
    )
    assert evaluation.passed, evaluation.failures


@pytest.mark.asyncio
async def test_genuine_failure_run_8_equivalent_is_ineligible() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    result = await execute_resolved_skeleton(fixture_bundle.bundle)
    filtered_nodes = _nodes_without_attendance(result.evidence_nodes)
    decision = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=filtered_nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    assert decision.status is CompletionEligibilityStatus.INELIGIBLE
    assert AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID in (
        decision.unresolved_mandatory_requirement_ids
    )


@pytest.mark.asyncio
async def test_genuine_failure_run_14_equivalent_is_ineligible() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    result = await execute_resolved_skeleton(fixture_bundle.bundle)
    filtered_nodes = _nodes_without_attendance(result.evidence_nodes)
    with pytest.raises(CompletionEligibilityBlockedError) as exc_info:
        assert_ai_incident_completion_eligible(
            evidence_nodes=filtered_nodes,
            gate=CompletionEligibilityGateConfig(),
        )
    assert AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID in (
        exc_info.value.decision.unresolved_mandatory_requirement_ids
    )


@pytest.mark.asyncio
async def test_proxy_only_low_tool_control_remains_eligible() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    result = await execute_resolved_skeleton(fixture_bundle.bundle)
    for tool_trace_count in (1, 2):
        decision = evaluate_ai_incident_completion_eligibility(
            evidence_nodes=result.evidence_nodes,
            provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
        )
        assert decision.status is CompletionEligibilityStatus.ELIGIBLE
        evaluation = evaluate_scenario_run(
            replace(result, tool_trace_count=tool_trace_count),
            fixture_bundle.fixture,
        )
        assert evaluation.passed, evaluation.failures


@pytest.mark.parametrize("tool_trace_count", (1, 2, 6))
@pytest.mark.asyncio
async def test_tool_count_invariance(tool_trace_count: int) -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    result = await execute_resolved_skeleton(fixture_bundle.bundle)
    baseline = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=result.evidence_nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    assert baseline.status is CompletionEligibilityStatus.ELIGIBLE
    _ = replace(result, tool_trace_count=tool_trace_count)
    variant = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=result.evidence_nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    assert variant.status == baseline.status
    assert variant.unresolved_mandatory_requirement_ids == (
        baseline.unresolved_mandatory_requirement_ids
    )


def test_planner_final_with_satisfied_requirements_is_eligible() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    nodes = (
        {"evidence_id": "evidence.comparison.line3.high_load_window", "payload": {}},
        {"evidence_id": "evidence.staffing.schedule.line4.incident_window", "payload": {}},
        {"evidence_id": "evidence.staffing.attendance.line4.incident_window", "payload": {}},
        {
            "evidence_id": "evidence.telemetry.complex_assembly_station.incident_window",
            "payload": {},
        },
    )
    decision = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    assert decision.status is CompletionEligibilityStatus.ELIGIBLE
    _ = fixture_bundle


def test_premature_final_with_missing_mandatory_is_ineligible() -> None:
    nodes = (
        {"evidence_id": "evidence.comparison.line3.high_load_window", "payload": {}},
        {"evidence_id": "evidence.staffing.schedule.line4.incident_window", "payload": {}},
        {
            "evidence_id": "evidence.telemetry.complex_assembly_station.incident_window",
            "payload": {},
        },
    )
    decision = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    assert decision.status is CompletionEligibilityStatus.INELIGIBLE
    assert AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID in (
        decision.unresolved_mandatory_requirement_ids
    )


def test_model_self_waiver_claim_does_not_waive_requirement() -> None:
    nodes = (
        {"evidence_id": "evidence.comparison.line3.high_load_window", "payload": {}},
        {"evidence_id": "evidence.staffing.schedule.line4.incident_window", "payload": {}},
    )
    decision = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    assert decision.status is CompletionEligibilityStatus.INELIGIBLE
    assert AI_INCIDENT_TELEMETRY_EVIDENCE_REQUIREMENT_ID in (
        decision.unresolved_mandatory_requirement_ids
    )


def test_provider_absence_fail_closed_when_gate_enabled() -> None:
    with pytest.raises(EvidenceRequirementProviderAbsentError):
        assert_ai_incident_completion_eligible(
            evidence_nodes=(),
            gate=CompletionEligibilityGateConfig(enabled=True, provider=None),
        )


def test_gate_disabled_skips_enforcement() -> None:
    assert (
        assert_ai_incident_completion_eligible(
            evidence_nodes=(),
            gate=CompletionEligibilityGateConfig(enabled=False),
        )
        is None
    )
