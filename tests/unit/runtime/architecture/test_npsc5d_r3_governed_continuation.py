# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R3 — canonical HITL governed physical delegation continuation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskAcquisitionError,
    DelegatedSubtaskContinuationGrantError,
    DelegatedSubtaskGovernanceDenied,
    DelegatedSubtaskGovernanceRequiresHuman,
    DelegatedSubtaskInvocation,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentError
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.execution.boundary import ExecutionBoundary
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
from tests.unit.agent_distribution.test_physical_delegation_governance_boundary import (
    _CountingSelector,
    _CountingTaskScopedAgents,
    _DelegationRequireHumanGovernance,
    _build_instrumented_harness,
)
from tests.unit.runtime.human.test_g5b_hitl_resolution import bound_hitl_test_execution_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

RUN_ID = mint_run_id()
SOURCE_AGENT = "agent-coordinator"
APPROVER = local_development_approver_evidence(tenant_id="tenant-a")


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


def test_fan_out_slot_continuation_seam_required() -> None:
    pytest.skip("NEXUS SLOT CONTINUATION SEAM REQUIRED for fan-out HITL resume E2E")


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
