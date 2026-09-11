# © Artur Czarnecki. All rights reserved.

"""DG-001 — multi-agent execution → durable lineage → central diagnostic read (P3)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import cast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.agent_distribution.multi_agent_coordination import (
    ChildExecutionFailedError,
)
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
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
)
from intergrax.runtime.diagnostics.execution_lineage_reconstruction import (
    ExecutionLineageReadStatus,
    reconstruct_attempt_lineage,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
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
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
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
from intergrax.contracts.delegation_authority import (
    resolve_root_parent_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from testing_support.agent_distribution.coordination_governance import (
    bound_governed_host_task,
)
from testing_support.agent_distribution.delegated_subtask_qualification_harness import (
    DelegatedSubtaskQualificationHarness,
    OcrQualificationRequest as OcrRequest,
    OcrQualificationResult as OcrResult,
    OCR_QUALIFICATION_PACKAGE_ID,
    build_ocr_qualification_discovery_candidate,
)
from testing_support.agent_distribution.decision_coordination_qualification import (
    DecisionCoordinationExecutorFixture,
    accepted_decision,
    build_decision_coordination_executor_fixture,
    coordination_binding,
    decision_contribution,
    project_accepted_decision,
)
from testing_support.agent_distribution.dg001_canonical_multi_agent_diagnostic_harness import (
    build_dg001_canonical_multi_agent_diagnostic_harness,
)
from testing_support.agent_distribution.multi_agent_coordination_qualification_harness import (
    qualification_root_execution_identity_binding,
)
from testing_support.agent_platform_admin_harness import admin_test_principal
from intergrax.runtime.diagnostics.persistence_conformance import (
    query_all_problems_for_tenant,
)
from intergrax.contracts.execution_identity import validate_run_id
from intergrax.runtime.task.task import Task, TaskContext, TaskState

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
    delegated_harness = cast(DelegatedSubtaskQualificationHarness, fixture.harness)
    delegated_harness.task_scope_authority.task_scope_id = task_scope
    binding = coordination_binding(task_scope, contribution_lease_pairs)
    root = qualification_root_execution_identity_binding()
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
            if (
                isinstance(node, ast.Name)
                and node.id in _FORBIDDEN_DIAGNOSTIC_AUTHORITY
            ):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                violations.append(f"{rel}:{node.lineno}:{node.id}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _FORBIDDEN_DIAGNOSTIC_AUTHORITY
            ):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                violations.append(f"{rel}:{node.lineno}.{node.attr}")
    assert violations == [], (
        "agent_distribution must not own diagnostic authority:\n"
        + "\n".join(
            violations,
        )
    )


@pytest.mark.asyncio
async def test_dg001_p3_root_single_child_forensic_topology_via_coordination_intent() -> (
    None
):
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
            build_ocr_qualification_discovery_candidate(
                OCR_QUALIFICATION_PACKAGE_ID,
                capability_ids=("document.ocr",),
            ),
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
            build_ocr_qualification_discovery_candidate(
                OCR_QUALIFICATION_PACKAGE_ID,
                capability_ids=("document.ocr",),
            ),
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
        for record in run.persistence.list_admissions_for_attempt(
            run.scope, limit=50
        ).admissions
        if record.parent_execution_id == run.root.execution_id
    ]
    assert len(positions) == 3
    assert len(set(positions)) == 3


@pytest.mark.asyncio
async def test_dg001_p3_nested_specialist_preserves_direct_forensic_parent_chain() -> (
    None
):
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
            return await nested_runner.execute(
                request=request, delegate=_LeafSpecialist()
            )

    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            build_ocr_qualification_discovery_candidate(
                OCR_QUALIFICATION_PACKAGE_ID,
                capability_ids=("document.ocr",),
            ),
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
            build_ocr_qualification_discovery_candidate(
                OCR_QUALIFICATION_PACKAGE_ID,
                capability_ids=("document.ocr",),
            ),
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
async def test_dg001_p3_canonical_root_clean_multi_agent_no_false_problem() -> None:
    tenant = f"{_TENANT}-canonical-clean"
    harness = build_dg001_canonical_multi_agent_diagnostic_harness(tenant_id=tenant)
    task = Task(
        tenant_id=tenant,
        user_id="user-dg001",
        message="canonical clean multi-agent",
        context=TaskContext(capability="dg001.multi_agent.coordination"),
        agent_id="dg001-multi-agent-root",
    )
    result = await harness.runner.run_task(task)
    assert result.state is TaskState.COMPLETED
    assert (
        query_all_problems_for_tenant(
            harness.diagnostic_dependencies.problem_persistence,
            tenant,
        )
        == ()
    )
    events = harness.runtime_event_store.list_for_task(
        str(task.task_id),
        tenant_id=tenant,
        limit=50,
    )
    assert events
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant,
        task_id=task.task_id,
        run_id=validate_run_id(result.run_id),
        attempt_id=events[0].attempt_id,
    )
    admissions = harness.lineage_persistence.list_admissions_for_attempt(
        scope, limit=50
    )
    assert len(admissions.admissions) >= 2


@pytest.mark.asyncio
async def test_dg001_p3_canonical_runtime_events_persisted_from_execution() -> None:
    tenant = f"{_TENANT}-canonical-events"
    harness = build_dg001_canonical_multi_agent_diagnostic_harness(tenant_id=tenant)
    task = Task(
        tenant_id=tenant,
        user_id="user-dg001",
        message="runtime evidence",
        context=TaskContext(capability="dg001.multi_agent.coordination"),
        agent_id="dg001-multi-agent-root",
    )
    result = await harness.runner.run_task(task)
    assert result.state is TaskState.COMPLETED
    assert result.run_id is not None
    events = harness.runtime_event_store.list_for_task(
        str(task.task_id),
        tenant_id=tenant,
        limit=200,
    )
    assert events
    assert any(
        event.event_type
        in {RuntimeEventType.TASK_CREATED, RuntimeEventType.TASK_COMPLETED}
        for event in events
    )


@pytest.mark.asyncio
async def test_dg001_p3_canonical_operator_read_after_terminal_trigger() -> None:
    tenant = f"{_TENANT}-canonical-read"
    harness = build_dg001_canonical_multi_agent_diagnostic_harness(tenant_id=tenant)
    task = Task(
        tenant_id=tenant,
        user_id="user-dg001",
        message="operator read",
        context=TaskContext(capability="dg001.multi_agent.coordination"),
        agent_id="dg001-multi-agent-root",
    )
    result = await harness.runner.run_task(task)
    assert result.state is TaskState.COMPLETED
    problems = harness.read_service.list_problems(tenant_id=tenant)
    assert problems.total_count == 0


@pytest.mark.asyncio
async def test_dg001_p3_real_child_failure_evidence_presence() -> None:
    class _FailingSpecialist:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            raise RuntimeError("controlled child failure")

    tenant = f"{_TENANT}-child-failure"
    harness = build_dg001_canonical_multi_agent_diagnostic_harness(
        tenant_id=tenant,
        specialist_delegate=_FailingSpecialist(),
    )
    task = Task(
        tenant_id=tenant,
        user_id="user-dg001",
        message="child failure",
        context=TaskContext(capability="dg001.multi_agent.coordination"),
        agent_id="dg001-multi-agent-root",
    )
    run_id: str | None = None
    try:
        result = await harness.runner.run_task(task)
        assert result.state is TaskState.FAILED
        run_id = result.run_id
    except ChildExecutionFailedError:
        events_probe = harness.runtime_event_store.list_for_task(
            str(task.task_id),
            tenant_id=tenant,
            limit=10,
        )
        assert events_probe
        run_id = str(events_probe[0].run_id)
    assert run_id is not None
    events = harness.runtime_event_store.list_for_task(
        str(task.task_id),
        tenant_id=tenant,
        limit=50,
    )
    assert events
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant,
        task_id=task.task_id,
        run_id=validate_run_id(run_id),
        attempt_id=events[0].attempt_id,
    )
    admissions = harness.lineage_persistence.list_admissions_for_attempt(
        scope, limit=50
    )
    assert len(admissions.admissions) >= 2


_DG001_CENTRAL_PROBLEM_BLOCKER = (
    "BLOCKED: Canonical multi-agent child failure reaches durable execution lineage, "
    "but current diagnostic evidence/terminal analysis does not produce a "
    "central Problem through the production diagnostic spine."
)


@pytest.mark.asyncio
async def test_dg001_p3_canonical_real_multi_agent_failure_central_problem_operator_read() -> (
    None
):
    """REAL MULTI-AGENT FAILURE → central Problem → operator read (production spine only)."""

    @dataclass
    class _FailureCapture:
        execution_id: ExecutionId | None = None

    capture = _FailureCapture()

    class _FailingSpecialist:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            capture.execution_id = require_active_execution_id()
            raise RuntimeError("controlled child failure")

    tenant = f"{_TENANT}-canonical-failure-central"
    harness = build_dg001_canonical_multi_agent_diagnostic_harness(
        tenant_id=tenant,
        specialist_delegate=_FailingSpecialist(),
    )
    task = Task(
        tenant_id=tenant,
        user_id="user-dg001",
        message="canonical failure central problem",
        context=TaskContext(capability="dg001.multi_agent.coordination"),
        agent_id="dg001-multi-agent-root",
    )
    run_id: str | None = None
    try:
        result = await harness.runner.run_task(task)
        assert result.state is TaskState.FAILED
        run_id = result.run_id
    except ChildExecutionFailedError:
        events_probe = harness.runtime_event_store.list_for_task(
            str(task.task_id),
            tenant_id=tenant,
            limit=10,
        )
        assert events_probe
        run_id = str(events_probe[0].run_id)

    assert run_id is not None
    validated_run_id = validate_run_id(run_id)
    events = harness.runtime_event_store.list_for_task(
        str(task.task_id),
        tenant_id=tenant,
        limit=50,
    )
    assert events
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant,
        task_id=task.task_id,
        run_id=validated_run_id,
        attempt_id=events[0].attempt_id,
    )
    admission_page = harness.lineage_persistence.list_admissions_for_attempt(
        scope, limit=50
    )
    assert len(admission_page.admissions) >= 2
    root_admissions = [
        record
        for record in admission_page.admissions
        if record.parent_execution_id is None
    ]
    child_admissions = [
        record
        for record in admission_page.admissions
        if record.parent_execution_id is not None
    ]
    assert root_admissions
    assert child_admissions
    root_execution_id = root_admissions[0].execution_id

    listed = harness.read_service.list_problems(tenant_id=tenant)
    assert listed.total_count >= 1
    problem_id = listed.problems[0].problem_id
    detail = harness.read_service.get_problem(
        tenant_id=tenant,
        problem_id=problem_id,
    )
    assert detail is not None
    assert detail.occurrence_count >= 1
    assert detail.occurrences

    matched_occurrence = False
    for occurrence_view in detail.occurrences:
        execution_subject = occurrence_view.subject_ref.execution()
        if execution_subject is None:
            continue
        if (
            execution_subject.task_id == task.task_id
            and execution_subject.run_id == validated_run_id
        ):
            matched_occurrence = True
        assert occurrence_view.read_status is DiagnosticOccurrenceReadStatus.AVAILABLE
        lineage_view = occurrence_view.execution_lineage
        assert lineage_view is not None
        assert lineage_view.attempts
        attempt_view = lineage_view.attempts[0]
        assert attempt_view.read_status is ExecutionLineageReadStatus.AVAILABLE
        projected_ids = {
            node.execution_id
            for segment in attempt_view.segments
            for node in segment.executions
        }
        assert root_execution_id in projected_ids
        assert any(
            child.execution_id in projected_ids for child in child_admissions
        )

    assert matched_occurrence, (
        "occurrence must reference the failing task/run scope"
    )

    assert capture.execution_id is not None
    failing_specialist_execution_id = capture.execution_id
    execution_failed_events = [
        event
        for event in events
        if event.event_type is RuntimeEventType.EXECUTION_FAILED
        and event.execution_id == failing_specialist_execution_id
    ]
    assert len(execution_failed_events) == 1
    failed_event = execution_failed_events[0]
    assert failed_event.tenant_id == tenant
    assert failed_event.task_id == task.task_id
    assert failed_event.run_id == validated_run_id
    assert failed_event.attempt_id == events[0].attempt_id
    from intergrax.runtime.events.payload_registry import validate_payload_envelope
    from intergrax.runtime.events.payloads.canonical import ExecutionFailurePayloadV1
    from intergrax.contracts.execution_failure_evidence import ExecutionFailureKind
    from intergrax.runtime.diagnostics.diagnostic_assessment import (
        DiagnosticFindingKind,
    )
    from intergrax.runtime.diagnostics.diagnostic_precision import (
        DiagnosticCertainty,
        DiagnosticPrecision,
    )

    payload = validate_payload_envelope(failed_event.payload)
    assert isinstance(payload, ExecutionFailurePayloadV1)
    assert payload.failure_kind is ExecutionFailureKind.DELEGATE_EXCEPTION
    assert "controlled child failure" not in payload.safe_summary
    assert payload.safe_summary

    proven_execution_finding = False
    for occurrence_view in detail.occurrences:
        assessment = occurrence_view.assessment
        if assessment is None:
            continue
        for finding in assessment.findings:
            if finding.kind is not DiagnosticFindingKind.EXECUTION_FAILED:
                continue
            if finding.execution_id != failing_specialist_execution_id:
                continue
            assert finding.certainty is DiagnosticCertainty.PROVEN
            assert finding.precision is DiagnosticPrecision.EXECUTION_LEVEL
            assert finding.failure_boundary is not None
            assert (
                finding.failure_boundary.execution_id
                == failing_specialist_execution_id
            )
            assert finding.supporting_event_ids == (failed_event.event_id,)
            proven_execution_finding = True
    assert proven_execution_finding, (
        "operator assessment must contain PROVEN execution-level failure "
        "for the exact failing specialist execution"
    )
    root_only_findings = [
        finding
        for occurrence_view in detail.occurrences
        if occurrence_view.assessment is not None
        for finding in occurrence_view.assessment.findings
        if finding.kind is DiagnosticFindingKind.EXECUTION_FAILED
        and finding.execution_id == root_execution_id
    ]
    assert not root_only_findings, (
        "root execution_id alone must not satisfy exact-child execution failure proof"
    )
