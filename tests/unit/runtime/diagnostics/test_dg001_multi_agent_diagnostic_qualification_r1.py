# © Artur Czarnecki. All rights reserved.

"""DG-001 — multi-agent execution → durable lineage → central diagnostic read (P3)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import cast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.decision_coordination import DecisionCoordinationShape
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    TaskId,
    bind_active_execution_identity,
    mint_task_id,
    peek_active_parent_execution_id,
    require_active_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptScope,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticExecutionScope,
    DiagnosticOrchestrationRequest,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageReadStatus,
    reconstruct_attempt_lineage,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    build_root_lineage_admission_hook,
    deactivate_root_execution_lineage,
    merge_lineage_root_admission_hooks,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.agent_distribution.coordination_governance import bound_governed_host_task
from testing_support.agent_distribution.decision_coordination_qualification import (
    DecisionCoordinationExecutorFixture,
    accepted_decision,
    build_decision_coordination_executor_fixture,
    coordination_binding,
    decision_contribution,
    project_accepted_decision,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    DelegatedHarness,
    OcrRequest,
    OcrResult,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import _root_identity
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
    read_service_for_tests,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-dg001-multi-agent"
_OBSERVED_AT = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())

_FORBIDDEN_DIAGNOSTIC_AUTHORITY = frozenset(
    {
        "DiagnosticOrchestrator",
        "DiagnosticReadService",
        "ProblemLifecycleEngine",
        "ProblemPersistence",
        "ProblemOccurrencePersistence",
        "ExecutionReconstructor",
    },
)


@dataclass(frozen=True, slots=True)
class Dg001DurableLineageRun:
    root: ExecutionIdentityBinding
    task_scope: TaskId
    scope: ExecutionLineageAttemptScope
    persistence: InMemoryExecutionLineagePersistence
    runtime_store: InMemoryRuntimeEventStore
    child_execution_ids: tuple[ExecutionId, ...]


def _admissions_by_execution(
    persistence: InMemoryExecutionLineagePersistence,
    scope: ExecutionLineageAttemptScope,
) -> dict[ExecutionId, ExecutionId | None]:
    page = persistence.list_admissions_for_attempt(scope, limit=500)
    return {
        record.execution_id: record.parent_execution_id for record in page.admissions
    }


async def _execute_coordination_with_durable_lineage(
    fixture: DecisionCoordinationExecutorFixture,
    *,
    accepted,
    contribution_lease_pairs: tuple[tuple[str, str], ...],
    bind_budget: bool = False,
) -> Dg001DurableLineageRun:
    intent = project_accepted_decision(accepted)
    task_scope = mint_task_id()
    delegated_harness = cast(DelegatedHarness, fixture.harness)
    delegated_harness.task_scope_authority.task_scope_id = task_scope
    binding = coordination_binding(task_scope, contribution_lease_pairs)
    root = _root_identity()
    scope = build_execution_lineage_attempt_scope(
        tenant_id=_TENANT,
        task_id=task_scope,
        run_id=root.run_id,
        attempt_id=root.attempt_id,
    )
    persistence = InMemoryExecutionLineagePersistence()
    runtime_store = InMemoryRuntimeEventStore()
    child_ids: list[ExecutionId] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            budget_token = None
            if bind_budget:
                budget_token = bind_root_execution_budget(
                    execution_id=require_active_execution_id(),
                    ledger=_UNLIMITED_LEDGER,
                )
            try:
                with bound_governed_host_task():
                    await fixture.executor.execute(
                        intent,
                        binding=binding,
                        principal=admin_test_principal(),
                    )
            finally:
                if budget_token is not None:
                    reset_active_execution_budget(budget_token)
            return OcrResult(text="root-done")

    _, lineage_token, degradation_token = activate_root_execution_lineage(
        persistence=persistence,
        scope=scope,
        root_execution_id=root.execution_id,
    )
    lineage_hook = build_root_lineage_admission_hook(
        persistence=persistence,
        scope=scope,
        segment_root_execution_id=root.execution_id,
        execution_id=root.execution_id,
    )
    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=root.run_id,
        attempt_id=root.attempt_id,
        execution_id=root.execution_id,
    )
    try:
        root_hooks = cast(
            tuple[ExecutionAdmissionHook[OcrRequest], ...],
            merge_lineage_root_admission_hooks(lineage_hook, ()),
        )
        await ExecutionBoundary[OcrRequest, OcrResult](
            RootDelegate(),
            admission_hooks=root_hooks,
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        ).execute(OcrRequest(document_ref="root"))
    finally:
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)
        deactivate_root_execution_lineage(lineage_token, degradation_token)

    page = persistence.list_admissions_for_attempt(scope, limit=500)
    child_ids.extend(
        record.execution_id
        for record in page.admissions
        if record.parent_execution_id == root.execution_id
    )
    return Dg001DurableLineageRun(
        root=root,
        task_scope=task_scope,
        scope=scope,
        persistence=persistence,
        runtime_store=runtime_store,
        child_execution_ids=tuple(child_ids),
    )


@pytest.mark.gate
def test_dg001_agent_distribution_does_not_instantiate_diagnostic_authority() -> None:
    root = _REPO_ROOT / "intergrax" / "agent_distribution"
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_DIAGNOSTIC_AUTHORITY:
                rel = path.relative_to(_REPO_ROOT).as_posix()
                violations.append(f"{rel}:{node.lineno}:{node.id}")
            if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_DIAGNOSTIC_AUTHORITY:
                rel = path.relative_to(_REPO_ROOT).as_posix()
                violations.append(f"{rel}:{node.lineno}.{node.attr}")
    assert violations == [], "agent_distribution must not own diagnostic authority:\n" + "\n".join(
        violations,
    )


@pytest.mark.asyncio
async def test_dg001_p3_root_single_child_forensic_topology_via_coordination_intent() -> None:
    specialist_child: ExecutionId | None = None
    specialist_parent: ExecutionId | None = None

    class _LineageSpecialist:
        async def execute(self, request: OcrRequest) -> OcrResult:
            nonlocal specialist_child, specialist_parent
            specialist_child = require_active_execution_id()
            specialist_parent = peek_active_parent_execution_id()
            return OcrResult(text=request.document_ref)

    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_LineageSpecialist(),
        fan_out=False,
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.SINGLE,
        (decision_contribution("contrib-a", document_ref="doc-single"),),
    )
    run = await _execute_coordination_with_durable_lineage(
        fixture,
        accepted=accepted,
        contribution_lease_pairs=(("contrib-a", "lease-a"),),
    )
    assert specialist_child is not None
    assert specialist_parent == run.root.execution_id
    by_exec = _admissions_by_execution(run.persistence, run.scope)
    assert by_exec[run.root.execution_id] is None
    assert by_exec[specialist_child] == run.root.execution_id
    lineage = reconstruct_attempt_lineage(
        run.persistence,
        tenant_id=run.scope.tenant_id,
        task_id=run.scope.task_id,
        run_id=run.scope.run_id,
        attempt_id=run.scope.attempt_id,
        initial_lineage_page_limit=100,
        max_lineage_records=10_000,
    )
    assert lineage.read_status is ExecutionLineageReadStatus.AVAILABLE
    assert len(lineage.segments) == 1
    assert lineage.segments[0].root_execution_id == run.root.execution_id


@pytest.mark.asyncio
async def test_dg001_p3_fan_out_three_siblings_share_parent_execution() -> None:
    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        fan_out=True,
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.FAN_OUT,
        (
            decision_contribution("contrib-a", document_ref="contrib-a"),
            decision_contribution("contrib-b", document_ref="contrib-b"),
            decision_contribution("contrib-c", document_ref="contrib-c"),
        ),
    )
    run = await _execute_coordination_with_durable_lineage(
        fixture,
        accepted=accepted,
        contribution_lease_pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
            ("contrib-c", "lease-c"),
        ),
        bind_budget=True,
    )
    assert len(run.child_execution_ids) == 3
    by_exec = _admissions_by_execution(run.persistence, run.scope)
    for child_id in run.child_execution_ids:
        assert by_exec[child_id] == run.root.execution_id
    positions = [
        record.admission_position
        for record in run.persistence.list_admissions_for_attempt(run.scope, limit=50).admissions
        if record.parent_execution_id == run.root.execution_id
    ]
    assert len(positions) == 3
    assert len(set(positions)) == 3


@pytest.mark.asyncio
async def test_dg001_p3_nested_specialist_preserves_direct_forensic_parent_chain() -> None:
    nested_child: ExecutionId | None = None
    nested_parent: ExecutionId | None = None
    mid_child: ExecutionId | None = None
    ledger = _UNLIMITED_LEDGER
    nested_runner = ChildExecutionRunner[OcrRequest, OcrResult](ledger=ledger)

    class _LeafSpecialist:
        async def execute(self, request: OcrRequest) -> OcrResult:
            nonlocal nested_child, nested_parent
            nested_child = require_active_execution_id()
            nested_parent = peek_active_parent_execution_id()
            return OcrResult(text=f"leaf:{request.document_ref}")

    class _MidSpecialist:
        async def execute(self, request: OcrRequest) -> OcrResult:
            nonlocal mid_child
            mid_child = require_active_execution_id()
            return await nested_runner.execute(request=request, delegate=_LeafSpecialist())

    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_MidSpecialist(),
        fan_out=False,
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.SINGLE,
        (decision_contribution("contrib-nested", document_ref="nested-doc"),),
    )
    run = await _execute_coordination_with_durable_lineage(
        fixture,
        accepted=accepted,
        contribution_lease_pairs=(("contrib-nested", "lease-nested"),),
        bind_budget=True,
    )
    assert mid_child is not None
    assert nested_child is not None
    assert nested_parent == mid_child
    by_exec = _admissions_by_execution(run.persistence, run.scope)
    assert by_exec[run.root.execution_id] is None
    assert by_exec[mid_child] == run.root.execution_id
    assert by_exec[nested_child] == mid_child


@pytest.mark.asyncio
async def test_dg001_p3_partial_sibling_failure_preserves_all_admissions() -> None:
    class _PartialFailureSpecialist:
        async def execute(self, request: OcrRequest) -> OcrResult:
            if request.document_ref == "contrib-b":
                raise RuntimeError("controlled specialist failure")
            return OcrResult(text=request.document_ref)

    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_PartialFailureSpecialist(),
        fan_out=True,
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.FAN_OUT,
        (
            decision_contribution("contrib-a", document_ref="contrib-a"),
            decision_contribution("contrib-b", document_ref="contrib-b"),
            decision_contribution("contrib-c", document_ref="contrib-c"),
        ),
    )
    run = await _execute_coordination_with_durable_lineage(
        fixture,
        accepted=accepted,
        contribution_lease_pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
            ("contrib-c", "lease-c"),
        ),
        bind_budget=True,
    )
    assert len(run.child_execution_ids) == 3
    by_exec = _admissions_by_execution(run.persistence, run.scope)
    for child_id in run.child_execution_ids:
        assert by_exec[child_id] == run.root.execution_id


@pytest.mark.asyncio
async def test_dg001_p3_operator_read_surfaces_multi_agent_lineage_after_orchestrator() -> None:
    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        fan_out=False,
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.SINGLE,
        (decision_contribution("contrib-ok", document_ref="ok-doc"),),
    )
    run = await _execute_coordination_with_durable_lineage(
        fixture,
        accepted=accepted,
        contribution_lease_pairs=(("contrib-ok", "lease-ok"),),
    )
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
        RuntimeEventType.RETRY_SCHEDULED,
    ):
        event = sample_runtime_event(
            tenant_id=run.scope.tenant_id,
            task_id=run.scope.task_id,
            run_id=run.scope.run_id,
            attempt_id=run.scope.attempt_id,
        ).model_copy(update={"event_type": event_type})
        run.runtime_store.append(event, tenant_id=run.scope.tenant_id)

    causal = InMemoryCausalEvidencePersistence()
    reconstructor = ExecutionReconstructor(
        runtime_events=run.runtime_store,
        causal_evidence=causal,
        execution_lineage=run.persistence,
    )
    problem_persistence = InMemoryProblemPersistence()
    occurrence_store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(occurrence_store)
    lifecycle = ProblemLifecycleEngine(problem_persistence, occurrence_persistence)
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=reconstructor,
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=ProblemGroupingEngine(registry),
        problem_lifecycle_engine=lifecycle,
    )
    scoped = orchestrator.run(
        DiagnosticOrchestrationRequest(
            tenant_id=run.scope.tenant_id,
            executions=(
                DiagnosticExecutionScope(
                    tenant_id=run.scope.tenant_id,
                    task_id=run.scope.task_id,
                    run_id=run.scope.run_id,
                ),
            ),
            grouping_strategy_id=STRATEGY_ID,
            observed_at=_OBSERVED_AT,
        ),
    )
    assert scoped.execution_results
    assert scoped.execution_results[0].assessment.has_findings
    read_service = read_service_for_tests(
        problem_persistence,
        reconstructor,
        occurrence_persistence=occurrence_persistence,
        document_store=occurrence_store,
    )
    problems = read_service.list_problems(tenant_id=run.scope.tenant_id)
    assert problems.total_count is not None and problems.total_count >= 1
    problem_id = problems.problems[0].problem_id
    detail = read_service.get_problem(tenant_id=run.scope.tenant_id, problem_id=problem_id)
    assert detail is not None
    assert detail.occurrences
    view = detail.occurrences[0]
    assert view.execution_lineage is not None
    assert view.execution_lineage.attempts
    attempt_view = view.execution_lineage.attempts[0]
    assert attempt_view.read_status is ExecutionLineageReadStatus.AVAILABLE
    child_ids = {str(eid) for eid in run.child_execution_ids}
    exposed_ids = {
        str(node.execution_id)
        for segment in attempt_view.segments
        for node in segment.executions
    }
    assert child_ids.issubset(exposed_ids)
