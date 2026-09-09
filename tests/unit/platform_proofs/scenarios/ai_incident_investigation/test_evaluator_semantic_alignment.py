# © Artur Czarnecki. All rights reserved.

"""Evaluator semantic alignment tests (DS-E2E-15B.1)."""

from __future__ import annotations

from dataclasses import replace

import pytest

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


@pytest.mark.asyncio
async def test_semantic_complete_passes_with_low_tool_trace_count() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    result = await execute_resolved_skeleton(fixture_bundle.bundle)
    evaluation = evaluate_scenario_run(
        replace(result, tool_trace_count=1),
        fixture_bundle.fixture,
    )
    assert evaluation.passed, evaluation.failures
    assert "tool_runtime_not_exercised" not in evaluation.failures


@pytest.mark.parametrize("tool_trace_count", (1, 2, 3, 6))
@pytest.mark.asyncio
async def test_evaluator_result_invariant_to_tool_count_when_semantics_unchanged(
    tool_trace_count: int,
) -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    base_result = await execute_resolved_skeleton(fixture_bundle.bundle)
    baseline = evaluate_scenario_run(base_result, fixture_bundle.fixture)
    variant = evaluate_scenario_run(
        replace(base_result, tool_trace_count=tool_trace_count),
        fixture_bundle.fixture,
    )
    assert variant.passed == baseline.passed
    assert variant.failures == baseline.failures
    assert variant.passed is True


@pytest.mark.asyncio
async def test_semantic_incomplete_fails_despite_high_tool_trace_count() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    base_result = await execute_resolved_skeleton(fixture_bundle.bundle)
    attendance_id = str(STAFFING_ATTENDANCE_EVIDENCE_ID)
    filtered_nodes = tuple(
        node
        for node in base_result.evidence_nodes
        if str(node.get("evidence_id")) != attendance_id
    )
    result = replace(base_result, evidence_nodes=filtered_nodes, tool_trace_count=6)
    evaluation = evaluate_scenario_run(result, fixture_bundle.fixture)
    assert evaluation.passed is False
    assert "staffing_attendance_not_gathered" in evaluation.failures


@pytest.mark.asyncio
async def test_revision_semantic_violation_fails_despite_tool_trace_count() -> None:
    fixture_bundle = build_fixture_runtime_bundle()
    base_result = await execute_resolved_skeleton(fixture_bundle.bundle)
    result = replace(base_result, revision_used_tools=False, tool_trace_count=4)
    evaluation = evaluate_scenario_run(result, fixture_bundle.fixture)
    assert evaluation.passed is False
    assert "follow_up_not_via_tools" in evaluation.failures
