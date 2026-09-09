# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R3 — canonical HITL governed physical delegation continuation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutId,
    FanOutItemStatus,
    FanOutRequest,
)
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSelectionProvenanceKind,
    DelegatedSubtaskAcquisitionError,
    DelegatedSubtaskContinuationGrantError,
    DelegatedSubtaskGovernanceDenied,
    DelegatedSubtaskGovernanceRequiresHuman,
    DelegatedSubtaskInvocation,
)
from intergrax.agent_distribution.multi_agent_coordination import CoordinationFailureCode
from intergrax.contracts.execution_identity import (
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotContinuationError,
    OrchestrationSlotContinuationRequest,
    OrchestrationSlotId,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.fan_out_orchestration_adapter import (
    FanOutCoordinationSlotExecutor,
    FanOutGovernedSlotContinuationContext,
    FanOutSlotPayload,
    build_fan_out_orchestration_port,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_submission_port,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentError
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.contracts.execution_interrupt import InterruptType
from intergrax.contracts.governed_continuation import ContinuationReason
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationContinuationApprovalGrant,
    grant_matches_physical_delegation_continuation,
    physical_delegation_governed_continuation_digest,
)
from intergrax.runtime.governance.physical_delegation_governance import (
    DenyingPhysicalDelegationGovernance,
    RequireHumanPhysicalDelegationGovernance,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanApprovalResolutionError, HumanPauseCoordinator
from intergrax.runtime.human.physical_delegation_continuation_grant import (
    PhysicalDelegationContinuationGrantCoordinator,
)
from intergrax.runtime.human.physical_delegation_governed_continuation_bridge import (
    apply_physical_delegation_governed_continuation_pause,
    project_physical_delegation_to_governed_continuation_request,
)
from intergrax.runtime.task.task import Task
from testing_support.agent_distribution.coordination_governance import (
    bound_governed_host_task,
    require_human_physical_delegation_governance,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _LEGAL_PACKAGE,
    _OCR_PACKAGE,
    _delegated_request,
    _discovery_candidate,
    _root_identity,
    _run_delegation,
    admin_test_principal,
    build_delegated_harness,
)
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _fan_out_item,
    build_fan_out_harness,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
    _root_identity,
)
from tests.unit.agent_distribution.test_physical_delegation_governance_boundary import (
    _CountingSelector,
    _CountingTaskScopedAgents,
    _DelegationRequireHumanGovernance,
    _build_instrumented_harness,
)
from tests.unit.runtime.human.test_g5b_hitl_resolution import (
    ATTEMPT_ID,
    bound_hitl_test_execution_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DELEGATED_SOURCE = _REPO_ROOT / "intergrax" / "agent_distribution" / "delegated_subtasks.py"

RUN_ID = mint_run_id()
SOURCE_AGENT = "agent-coordinator"
APPROVER = local_development_approver_evidence(tenant_id="tenant-a")
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


def _fan_out_root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        execution_id=mint_execution_id(),
    )


@contextmanager
def _bound_task_scope_execution(*, task_id: str) -> Iterator[Task]:
    with bound_governed_host_task(
        Task(
            tenant_id="tenant-a",
            user_id="user-a",
            agent_id=SOURCE_AGENT,
            task_id=task_id,
        ),
    ) as task:
        with bound_hitl_test_execution_identity(run_id=RUN_ID):
            yield task


def _approve_physical_continuation(
    task: Task,
    continuation,
    *,
    run_id: str = RUN_ID,
) -> PhysicalDelegationContinuationApprovalGrant:
    apply_physical_delegation_governed_continuation_pause(
        task,
        continuation,
        source_agent_id=SOURCE_AGENT,
        run_id=run_id,
    )
    pause_record = task.runtime.governance.pause_record
    human_request = task.runtime.governance.human_request
    assert pause_record is not None
    assert human_request is not None
    assert task.runtime.governance.execution_interrupt is not None
    assert (
        task.runtime.governance.execution_interrupt.interrupt_type
        is InterruptType.HUMAN_JUDGMENT_REQUIRED
    )
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=APPROVER,
        pause_id=pause_record.pause_id,
        human_request_id=human_request.request_id,
        run_id=run_id,
    )
    grant = PhysicalDelegationContinuationGrantCoordinator.create_grant_from_approval(task)
    assert grant is not None
    return grant


async def _continue_governed_delegation(
    harness,
    *,
    task_scope,
    task: Task,
    continuation,
    grant: PhysicalDelegationContinuationApprovalGrant,
):
    root = _root_identity()
    captured: list[object] = []

    class _ResumeDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=request),
                continuation=continuation,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant.grant_id,
            )
            captured.append(result)
            return result.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        _ResumeDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="doc-1"))
    assert captured
    return captured[0]


async def _require_human_continuation(harness, *, task_scope) -> object:
    with pytest.raises(DelegatedSubtaskGovernanceRequiresHuman) as exc_info:
        await _run_delegation(harness, task_scope=task_scope)
    return exc_info.value.continuation


@pytest.mark.asyncio
async def test_projection_preserves_physical_delegation_identity() -> None:
    harness, _, _, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    request = project_physical_delegation_to_governed_continuation_request(
        continuation,
        source_agent_id=SOURCE_AGENT,
        run_id=RUN_ID,
    )
    assert request.reason is ContinuationReason.COMPLIANCE
    assert request.task_id == str(task_scope)
    assert request.run_id == RUN_ID
    assert request.source_agent_id == SOURCE_AGENT
    assert continuation.delegation_id in request.operation_id
    assert request.side_effect_scope_id is None


@pytest.mark.asyncio
async def test_require_human_canonical_pause_no_acquire() -> None:
    harness, _, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        _approve_physical_continuation(task, continuation)
        assert task.runtime.governance.paused is True
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_single_e2e_resume_exact_specialist() -> None:
    harness, selector, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=_DelegationRequireHumanGovernance(
            require_human_delegation_id="delegation-1",
        ),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    assert continuation.selected_identity.distribution_package_id == _OCR_PACKAGE
    assert selector.call_count == 1

    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        result = await _continue_governed_delegation(
            harness,
            task_scope=task_scope,
            task=task,
            continuation=continuation,
            grant=grant,
        )
        assert result.result.text == "ocr:doc-1"
        assert result.selected_identity.package.distribution_package_id == _OCR_PACKAGE

    assert selector.call_count == 1
    assert task_scoped.acquire_count == 1
    assert specialist.call_count == 1
    assert child.call_count == 1
    assert task_scoped.release_count == 1
    assert (
        result.selection_provenance_kind
        is DelegatedSelectionProvenanceKind.PRESERVED_GOVERNED_CONTINUATION
    )
    assert result.selection_decision is None


def test_no_synthetic_governed_continuation_strategy_id() -> None:
    source = _DELEGATED_SOURCE.read_text(encoding="utf-8")
    assert "physical_delegation.governed_continuation" not in source


@pytest.mark.asyncio
async def test_no_discovery_on_resume() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        physical_delegation_governance=_DelegationRequireHumanGovernance(
            require_human_delegation_id="delegation-1",
        ),
    )
    discovery = harness.service._discovery
    matcher = harness.service._matcher
    selector = _CountingSelector(harness.service._selector)
    harness.service._selector = selector
    discovery_count = 0
    matcher_count = 0
    original_discover = discovery.discover
    original_find = matcher.find_matches

    def _counting_discover(*args, **kwargs):
        nonlocal discovery_count
        discovery_count += 1
        return original_discover(*args, **kwargs)

    def _counting_find(*args, **kwargs):
        nonlocal matcher_count
        matcher_count += 1
        return original_find(*args, **kwargs)

    discovery.discover = _counting_discover  # type: ignore[method-assign]
    matcher.find_matches = _counting_find  # type: ignore[method-assign]

    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    assert discovery_count == 1
    assert matcher_count == 1
    assert selector.call_count == 1

    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        await _continue_governed_delegation(
            harness,
            task_scope=task_scope,
            task=task,
            continuation=continuation,
            grant=grant,
        )

    assert discovery_count == 1
    assert matcher_count == 1
    assert selector.call_count == 1


@pytest.mark.asyncio
async def test_wrong_grant_id_fail_closed() -> None:
    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        _approve_physical_continuation(task, continuation)
        with pytest.raises(DelegatedSubtaskContinuationGrantError):
            await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=continuation,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id="wrong-grant",
            )
    assert task_scoped.acquire_count == 0


@pytest.mark.asyncio
async def test_grant_at_most_once() -> None:
    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        await _continue_governed_delegation(
            harness,
            task_scope=task_scope,
            task=task,
            continuation=continuation,
            grant=grant,
        )
        with pytest.raises(DelegatedSubtaskContinuationGrantError):
            await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=continuation,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant.grant_id,
            )
    assert task_scoped.acquire_count == 1


@pytest.mark.asyncio
async def test_reject_no_acquire() -> None:
    harness, _, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        apply_physical_delegation_governed_continuation_pause(
            task,
            continuation,
            source_agent_id=SOURCE_AGENT,
            run_id=RUN_ID,
        )
        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None
        assert human_request is not None
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.REJECT,
            approver=APPROVER,
            pause_id=pause_record.pause_id,
            human_request_id=human_request.request_id,
            run_id=RUN_ID,
        )
        assert (
            PhysicalDelegationContinuationGrantCoordinator.create_grant_from_approval(task)
            is None
        )
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_policy_deny_after_approval() -> None:
    class _FlipDenyGovernance:
        def __init__(self) -> None:
            self._calls = 0

        def evaluate(self, request):
            self._calls += 1
            if self._calls == 1:
                return RequireHumanPhysicalDelegationGovernance().evaluate(request)
            return DenyingPhysicalDelegationGovernance().evaluate(request)

    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=_FlipDenyGovernance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        with pytest.raises(DelegatedSubtaskGovernanceDenied):
            await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=continuation,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant.grant_id,
            )
    assert task_scoped.acquire_count == 0


@pytest.mark.asyncio
async def test_ac3_deny_after_approval() -> None:
    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    inner = harness.task_scoped
    counting = _CountingTaskScopedAgents(inner)
    harness.service._task_scoped_agents = counting
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)

    def _deny_acquire(*args, **kwargs):
        raise TaskScopedAgentError("ac-3 deny")

    counting.acquire = _deny_acquire  # type: ignore[method-assign]

    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        with pytest.raises(DelegatedSubtaskAcquisitionError):
            await _continue_governed_delegation(
                harness,
                task_scope=task_scope,
                task=task,
                continuation=continuation,
                grant=grant,
            )
    assert counting.acquire_count == 0


@pytest.mark.asyncio
async def test_grant_creation_wrong_pause_fail_closed() -> None:
    harness, _, _, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        apply_physical_delegation_governed_continuation_pause(
            task,
            continuation,
            source_agent_id=SOURCE_AGENT,
            run_id=RUN_ID,
        )
        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None
        assert human_request is not None
        with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
            HumanPauseCoordinator.resolve_human_response(
                task,
                HumanResponseVerdict.APPROVE,
                approver=APPROVER,
                pause_id="wrong-pause",
                human_request_id=human_request.request_id,
                run_id=RUN_ID,
            )


def _build_governed_fan_out_stack(candidates):
    harness = build_fan_out_harness(
        candidates=candidates,
        physical_delegation_governance=_DelegationRequireHumanGovernance(
            require_human_delegation_id="delegation-b",
        ),
    )
    selector = _CountingSelector(harness.service._selector)
    task_scoped = _CountingTaskScopedAgents(harness.task_scoped)
    from tests.unit.agent_distribution.test_physical_delegation_governance_boundary import (
        _CountingChildExecution,
        _CountingSpecialistInvocation,
    )

    specialist = _CountingSpecialistInvocation(harness.service._specialist_invocation)
    child = _CountingChildExecution(harness.service._child_execution)
    harness.service._selector = selector
    harness.service._task_scoped_agents = task_scoped
    harness.service._specialist_invocation = specialist
    harness.service._child_execution = child
    coordination = _build_coordination_service(harness)
    nexus_loop = NexusLoop(AgentRegistry())
    topology_port = build_orchestration_topology_submission_port(nexus_loop)
    adapter = build_fan_out_orchestration_port(topology_port, coordination)
    fan_out = BoundedMultiAgentFanOutService(orchestration=adapter)
    return harness, selector, task_scoped, specialist, child, adapter, fan_out


async def _run_governed_fan_out(
    harness,
    fan_out,
    *,
    task_scope,
    items,
    fan_out_id: str = "fan-out-abc",
):
    harness.task_scope_authority.task_scope_id = task_scope
    root = _fan_out_root_identity()
    captured = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=_UNLIMITED_LEDGER,
            )
            try:
                result = await fan_out.fan_out(
                    FanOutRequest(
                        fan_out_id=FanOutId(fan_out_id),
                        items=items,
                        max_concurrency=3,
                    ),
                    principal=admin_test_principal(),
                )
            finally:
                reset_active_execution_budget(budget_token)
            captured.append(result)
            return OcrResult(text="root-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    assert captured
    return captured[0]


@pytest.mark.asyncio
async def test_fan_out_e2e_resume_exact_blocked_slot_without_sibling_rerun() -> None:
    candidates = (
        _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
    )
    harness, selector, task_scoped, _, child, adapter, fan_out = _build_governed_fan_out_stack(
        candidates,
    )
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
        _fan_out_item(
            item_id="c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
            document_ref="doc-c",
        ),
    )
    initial_result = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
    )
    initial = initial_result.items
    assert len(initial) == 3
    assert [outcome.item_id for outcome in initial] == [item.item_id for item in items]
    assert initial[0].status is FanOutItemStatus.SUCCESS
    assert initial[1].status is FanOutItemStatus.FAILURE
    assert initial[1].failure is not None
    assert (
        initial[1].failure.failure_code
        is CoordinationFailureCode.GOVERNANCE_REQUIRES_HUMAN
    )
    assert initial[1].failure.continuation is not None
    assert initial[2].status is FanOutItemStatus.SUCCESS
    assert child.call_count == 2
    assert selector.call_count == 3

    continuation = initial[1].failure.continuation
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        resumed = await _run_governed_fan_out_resume(
            harness,
            adapter,
            task_scope=task_scope,
            items=items,
            item_id=items[1].item_id,
            continuation=continuation,
            grant=grant,
            task=task,
        )

    assert len(resumed) == 3
    assert resumed[0].status is FanOutItemStatus.SUCCESS
    assert resumed[0].result is not None
    assert resumed[0].result.result.text == "ocr:doc-a"
    assert resumed[1].status is FanOutItemStatus.SUCCESS
    assert resumed[1].result is not None
    assert resumed[1].result.result.text == "ocr:doc-b"
    assert resumed[2].status is FanOutItemStatus.SUCCESS
    assert resumed[2].result is not None
    assert resumed[2].result.result.text == "ocr:doc-c"
    assert child.call_count == 3
    assert selector.call_count == 3
    assert task_scoped.acquire_count == 3


async def _run_governed_fan_out_resume(
    harness,
    adapter,
    *,
    task_scope,
    items,
    item_id,
    continuation,
    grant,
    task: Task,
    correlation_id: str = "resume-b-1",
):
    harness.task_scope_authority.task_scope_id = task_scope
    root = _fan_out_root_identity()
    request = FanOutRequest(
        fan_out_id=FanOutId("fan-out-abc"),
        items=items,
        max_concurrency=3,
    )
    captured = []

    class RootDelegate:
        async def execute(self, request_payload: OcrRequest) -> OcrResult:
            del request_payload
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=_UNLIMITED_LEDGER,
            )
            try:
                result = await adapter.continue_governed_fan_out_slot(
                    request,
                    principal=admin_test_principal(),
                    item_id=item_id,
                    continuation_context=FanOutGovernedSlotContinuationContext(
                        continuation=continuation,
                        expected_grant_id=grant.grant_id,
                        task=task,
                    ),
                    correlation_id=correlation_id,
                )
            finally:
                reset_active_execution_budget(budget_token)
            captured.append(result)
            return OcrResult(text="root-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    assert captured
    return captured[0]


@pytest.mark.asyncio
async def test_fan_out_wrong_slot_continuation_blocked() -> None:
    candidates = (_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),)
    harness, _, _, _, _, adapter, fan_out = _build_governed_fan_out_stack(candidates)
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
        ),
        _fan_out_item(
            item_id="b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
        ),
        _fan_out_item(
            item_id="c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
        ),
    )
    initial_result = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
    )
    continuation = initial_result.items[1].failure.continuation
    assert continuation is not None
    request = FanOutRequest(
        fan_out_id=FanOutId("fan-out-abc"),
        items=items,
        max_concurrency=3,
    )
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        coordination = _build_coordination_service(harness)
        slot_executor = FanOutCoordinationSlotExecutor(
            coordination=coordination,
            principal=admin_test_principal(),
            resume_context=FanOutGovernedSlotContinuationContext(
                continuation=continuation,
                expected_grant_id=grant.grant_id,
                task=task,
            ),
        )
        root = _fan_out_root_identity()

        class WrongSlotDelegate:
            async def execute(self, request_payload: OcrRequest) -> OcrResult:
                del request_payload
                await adapter.topology_continuation.continue_slot(
                    OrchestrationSlotContinuationRequest(
                        execution_id=adapter.resolve_execution_id(
                            request,
                            principal=admin_test_principal(),
                        ),
                        slot_id=OrchestrationSlotId("c"),
                        correlation_id="wrong-slot",
                    ),
                    slot_continuation_executor=slot_executor,
                )
                return OcrResult(text="root-done")

        with pytest.raises(
            OrchestrationSlotContinuationError,
            match="terminal successful slot cannot be continued",
        ):
            await ExecutionBoundary[OcrRequest, OcrResult](
                WrongSlotDelegate(),
                identity=root,
                authority=ParentExecutionAuthority.unrestricted_root(),
            ).execute(OcrRequest(document_ref="root"))


@pytest.mark.asyncio
async def test_fan_out_duplicate_resume_blocked() -> None:
    candidates = (_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),)
    harness, _, task_scoped, _, child, adapter, fan_out = _build_governed_fan_out_stack(
        candidates,
    )
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
        _fan_out_item(
            item_id="c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
            document_ref="doc-c",
        ),
    )
    initial_result = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
    )
    continuation = initial_result.items[1].failure.continuation
    assert continuation is not None
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        await _run_governed_fan_out_resume(
            harness,
            adapter,
            task_scope=task_scope,
            items=items,
            item_id=items[1].item_id,
            continuation=continuation,
            grant=grant,
            task=task,
            correlation_id="resume-once",
        )
        with pytest.raises(
            OrchestrationSlotContinuationError,
            match="terminal successful slot cannot be continued",
        ):
            await _run_governed_fan_out_resume(
                harness,
                adapter,
                task_scope=task_scope,
                items=items,
                item_id=items[1].item_id,
                continuation=continuation,
                grant=grant,
                task=task,
                correlation_id="resume-twice",
            )
    assert child.call_count == 3
    assert task_scoped.acquire_count == 3


@pytest.mark.asyncio
async def test_grant_matches_wrong_delegation_blocked() -> None:
    harness, _, _, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _require_human_continuation(harness, task_scope=task_scope)
    wrong = continuation.model_copy(update={"delegation_id": "delegation-other"})
    grant = PhysicalDelegationContinuationApprovalGrant(
        grant_id="pdcg_test",
        continuation_digest=physical_delegation_governed_continuation_digest(continuation),
        continuation_request_id="gcr_test",
        delegation_id=continuation.delegation_id,
        task_scope_id=continuation.task_scope_id,
        run_id=RUN_ID,
        selected_identity=continuation.selected_identity,
        capability_requirement=continuation.capability_requirement,
        governance_request_digest=continuation.governance_result.evidence.request_digest,
        pause_id="pause-1",
        human_request_id="hr-1",
        approved_at="2026-09-09T12:00:00+00:00",
    )
    assert not grant_matches_physical_delegation_continuation(grant, wrong)
