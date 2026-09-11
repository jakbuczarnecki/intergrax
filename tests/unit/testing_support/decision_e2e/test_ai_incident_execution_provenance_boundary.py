# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-QI2.R1 typed execution provenance boundary proofs."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.contracts.execution_identity import mint_run_id
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    CompletionReconciliationError,
    reconcile_investigation_completion,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
    PreReconciliationTransitionDecision,
    PreReconciliationTransitionOutcome,
    PreReconciliationValidationError,
    PreReconciliationRecoveryStatus,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    ScenarioExecutionResult,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_execution_provenance import (
    ScenarioExecutionProvenance,
    scenario_execution_provenance,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_UNRESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
)
from testing_support.decision_e2e.ai_incident_qualification_run import (
    execute_ai_incident_qualification_run,
)


def test_scenario_execution_result_has_typed_provenance_only() -> None:
    names = {field.name for field in fields(ScenarioExecutionResult)}
    assert "execution_provenance" in names
    assert "persisted_trace_events" not in names
    assert "platform_run_id" not in names
    provenance_field = next(f for f in fields(ScenarioExecutionResult) if f.name == "execution_provenance")
    assert provenance_field.type in (
        ScenarioExecutionProvenance | None,
        "ScenarioExecutionProvenance | None",
    )


def test_semantic_exceptions_carry_provenance_not_raw_trace() -> None:
    for exc_type in (PreReconciliationValidationError, CompletionReconciliationError):
        assert "execution_provenance" in exc_type.__annotations__
        assert "persisted_trace_events" not in exc_type.__annotations__
        assert "platform_run_id" not in exc_type.__annotations__


def test_scenario_execution_provenance_is_immutable() -> None:
    run_id = mint_run_id()
    provenance = scenario_execution_provenance(str(run_id), "tenant-a")
    assert provenance.platform_run_id == run_id
    assert provenance.execution_tenant_id == "tenant-a"
    with pytest.raises((AttributeError, TypeError)):
        provenance.execution_tenant_id = "other"  # type: ignore[misc]


def test_qualification_run_module_does_not_use_list_runs_heuristic() -> None:
    from testing_support.decision_e2e import ai_incident_qualification_run as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "list_runs":
                pytest.fail("list_runs must not be used on canonical qualification trace path")


@pytest.mark.asyncio
async def test_success_path_reads_trace_via_exact_run_identity() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-trace"
    provenance = scenario_execution_provenance(str(run_id), tenant_id)
    result = MagicMock()
    result.execution_provenance = provenance
    result.execution_tenant_id = tenant_id
    result.planner_decisions = ()
    result.evaluator_loop_iterations = 0
    result.tool_invocations = 0
    result.evidence_nodes = ()
    result.initial_evidence_nodes = ()
    result.evidence_gathering_stop_reason = ""
    result.outcome = "resolved"
    result.investigation_conclusion = None
    result.critic_verdict_passed = True
    result.tool_execution_order = ()

    reader = MagicMock()
    reader.read_run.return_value = MagicMock(events=())
    composition = MagicMock()
    binding = object()
    bundle = MagicMock()
    bundle.bundle.runtime_composition = composition
    bundle.fixture = MagicMock()
    evaluation = MagicMock(passed=True, failures=())

    with (
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.bind_qualification_llm_profile",
            return_value=(binding, None),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.build_fixture_runtime_bundle",
            return_value=bundle,
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_canonical_runtime_modules",
            return_value=("decision_flow",),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_scenario_llm_adapter",
        ) as adapter_patch,
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.execute_resolved_skeleton",
            new=AsyncMock(return_value=result),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.evaluate_scenario_run",
            return_value=evaluation,
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.trace_reader_from_composition",
            return_value=reader,
        ),
    ):
        adapter = adapter_patch.return_value
        adapter.supports_strict_tool_argument_conformance.return_value = True
        outcome = await execute_ai_incident_qualification_run(run_index=0)

    reader.read_run.assert_called_once_with(str(run_id), tenant_id)
    assert outcome.runtime_execution_run_id == str(run_id)
    assert outcome.trace_evidence is not None
    assert outcome.trace_evidence.trace_correlation.value == "pass"


@pytest.mark.asyncio
async def test_pre_reconciliation_exception_reads_trace_from_typed_provenance() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-exc"
    exc = PreReconciliationValidationError(
        PreReconciliationTransitionDecision(
            outcome=PreReconciliationTransitionOutcome.REJECTED,
            validation_errors=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
            recovery_status=PreReconciliationRecoveryStatus.BUDGET_EXHAUSTED,
            revision_budget_remaining=0,
            completion_mode=COMPLETION_UNRESOLVED,
            has_supported_diagnosis=True,
            recovery_attempted=False,
        )
    )
    exc.execution_provenance = scenario_execution_provenance(str(run_id), tenant_id)

    reader = MagicMock()
    reader.read_run.return_value = MagicMock(events=({"event": "x"},))
    composition = MagicMock()
    binding = object()
    bundle = MagicMock()
    bundle.bundle.runtime_composition = composition
    bundle.fixture = MagicMock()

    with (
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.bind_qualification_llm_profile",
            return_value=(binding, None),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.build_fixture_runtime_bundle",
            return_value=bundle,
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_canonical_runtime_modules",
            return_value=("decision_flow",),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_scenario_llm_adapter",
        ) as adapter_patch,
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.execute_resolved_skeleton",
            new=AsyncMock(side_effect=exc),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.trace_reader_from_composition",
            return_value=reader,
        ),
    ):
        adapter = adapter_patch.return_value
        adapter.supports_strict_tool_argument_conformance.return_value = True
        outcome = await execute_ai_incident_qualification_run(run_index=0)

    reader.read_run.assert_called_once_with(str(run_id), tenant_id)
    assert outcome.trace_evidence is not None
    assert len(outcome.trace_evidence.trace_events) == 1


@pytest.mark.asyncio
async def test_reconciliation_exception_reads_trace_from_typed_provenance() -> None:
    run_id = mint_run_id()
    tenant_id = "tenant-recon"
    with pytest.raises(CompletionReconciliationError) as raised:
        reconcile_investigation_completion(
            model_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            validation_errors=("staffing_attendance_not_gathered",),
            evidence_gathering_stop_reason="planner_final_answer",
        )
    exc = raised.value
    exc.execution_provenance = scenario_execution_provenance(str(run_id), tenant_id)

    reader = MagicMock()
    reader.read_run.return_value = MagicMock(events=())
    composition = MagicMock()
    binding = object()
    bundle = MagicMock()
    bundle.bundle.runtime_composition = composition
    bundle.fixture = MagicMock()

    with (
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.bind_qualification_llm_profile",
            return_value=(binding, None),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.build_fixture_runtime_bundle",
            return_value=bundle,
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_canonical_runtime_modules",
            return_value=("decision_flow",),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.resolve_scenario_llm_adapter",
        ) as adapter_patch,
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.execute_resolved_skeleton",
            new=AsyncMock(side_effect=exc),
        ),
        patch(
            "testing_support.decision_e2e.ai_incident_qualification_run.trace_reader_from_composition",
            return_value=reader,
        ),
    ):
        adapter = adapter_patch.return_value
        adapter.supports_strict_tool_argument_conformance.return_value = True
        await execute_ai_incident_qualification_run(run_index=0)

    reader.read_run.assert_called_once_with(str(run_id), tenant_id)
