# © Artur Czarnecki. All rights reserved.

"""NPSC-5A — multi-agent coordination contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryContractError,
    AgentDiscoveryRequest,
    AgentDiscoveryResult,
    AgentDiscoveryStrategy,
    AgentDiscoveryStrategyId,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    AcquisitionFailedError,
    AuthorityScopeMismatchError,
    CapabilityResolutionFailedError,
    ChildExecutionFailedError,
    CoordinationDelegation,
    CoordinationFailureCode,
    CoordinationId,
    CoordinationRequest,
    CoordinationResult,
    MultiAgentCoordinationService,
    NoEligibleSpecialistError,
    validate_coordination_id,
)
from intergrax.agent_distribution.task_capability_resolution import (
    build_task_capability_resolution_request,
    unresolved_agent_distribution_capability_need,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLeaseId,
    TaskScopedAgentLeaseState,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.delegated_subtask_child_port import (
    as_child_execution_port,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _APP,
    _ENV,
    _LEGAL_PACKAGE,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
    build_delegated_harness,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


def _coordination_request(
    *,
    task_scope: TaskId,
    coordination_id: str = "coordination-1",
    delegation_id: str = "delegation-1",
    lease_id: str = "lease-delegate-1",
    task_kind: str = "document.ocr",
) -> CoordinationRequest:
    return CoordinationRequest(
        coordination_id=CoordinationId(coordination_id),
        delegation_id=delegation_id,
        task_scope_id=task_scope,
        application_id=_APP,
        application_environment_id=_ENV,
        lease_id=TaskScopedAgentLeaseId(lease_id),
        capability_need=unresolved_agent_distribution_capability_need(
            build_task_capability_resolution_request(task_kind=task_kind),
        ),
    )


def _build_coordination_service(
    harness,
) -> MultiAgentCoordinationService[OcrRequest, OcrResult]:
    return MultiAgentCoordinationService(
        delegated_subtasks=harness.service,
    )


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


async def _run_coordination(
    harness,
    *,
    task_scope: TaskId,
    document_ref: str = "doc-1",
    authority: ParentExecutionAuthority | None = None,
    coordination_id: str = "coordination-1",
    delegation_id: str = "delegation-1",
    lease_id: str = "lease-delegate-1",
) -> CoordinationResult[OcrResult]:
    coordination = _build_coordination_service(harness)
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()
    captured: list[CoordinationResult[OcrResult]] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await coordination.coordinate(
                _coordination_request(
                    task_scope=task_scope,
                    coordination_id=coordination_id,
                    delegation_id=delegation_id,
                    lease_id=lease_id,
                ),
                delegation=CoordinationDelegation(payload=request),
                principal=admin_test_principal(),
            )
            captured.append(result)
            return result.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=authority or ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref=document_ref))
    assert captured
    return captured[0]


def test_coordination_request_rejects_empty_coordination_id() -> None:
    task_scope = mint_task_id()
    with pytest.raises(ValidationError):
        CoordinationRequest(
            coordination_id=CoordinationId("   "),
            delegation_id="delegation-1",
            task_scope_id=task_scope,
            application_id=_APP,
            application_environment_id=_ENV,
            lease_id=TaskScopedAgentLeaseId("lease-1"),
            capability_need=unresolved_agent_distribution_capability_need(
                build_task_capability_resolution_request(task_kind="document.ocr"),
            ),
        )


def test_coordination_request_rejects_invalid_delegation_id() -> None:
    task_scope = mint_task_id()
    with pytest.raises(ValidationError):
        CoordinationRequest(
            coordination_id=CoordinationId("coordination-1"),
            delegation_id="",
            task_scope_id=task_scope,
            application_id=_APP,
            application_environment_id=_ENV,
            lease_id=TaskScopedAgentLeaseId("lease-1"),
            capability_need=unresolved_agent_distribution_capability_need(
                build_task_capability_resolution_request(task_kind="document.ocr"),
            ),
        )


def test_validate_coordination_id_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="coordination_id must be str"):
        validate_coordination_id(42)


@pytest.mark.asyncio
async def test_coordination_selects_eligible_specialist() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(
                _LEGAL_PACKAGE,
                capability_ids=("legal.analysis", "document.read"),
            ),
        ),
    )
    task_scope = mint_task_id()
    result = await _run_coordination(harness, task_scope=task_scope)
    assert result.coordination_id == CoordinationId("coordination-1")
    assert result.result.text == "ocr:doc-1"
    assert result.delegated.selected_identity.package.distribution_package_id == _OCR_PACKAGE


@pytest.mark.asyncio
async def test_coordination_no_eligible_specialist_fails_closed() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(
                _LEGAL_PACKAGE,
                capability_ids=("legal.analysis",),
            ),
        ),
    )
    coordination = _build_coordination_service(harness)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(NoEligibleSpecialistError) as exc_info:
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=request),
                    principal=admin_test_principal(),
                )
            assert (
                exc_info.value.failure_code
                is CoordinationFailureCode.NO_ELIGIBLE_SPECIALIST
            )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="none"))


class _FailingDiscovery(AgentDiscoveryStrategy):
    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return AgentDiscoveryStrategyId(value="failing.test")

    def discover(self, request: AgentDiscoveryRequest) -> AgentDiscoveryResult:
        del request
        raise AgentDiscoveryContractError("discovery failed")


@pytest.mark.asyncio
async def test_coordination_strategy_injection_preserved() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    harness.service._discovery = _FailingDiscovery()
    coordination = _build_coordination_service(harness)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(CapabilityResolutionFailedError):
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="discover"))


@pytest.mark.asyncio
async def test_coordination_task_scope_mismatch_rejected() -> None:
    canonical_task = mint_task_id()
    caller_task = mint_task_id()
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        task_scope=canonical_task,
    )
    coordination = _build_coordination_service(harness)
    root = _root_identity()
    harness.task_scope_authority.task_scope_id = canonical_task

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(AuthorityScopeMismatchError) as exc_info:
                await coordination.coordinate(
                    _coordination_request(task_scope=caller_task),
                    delegation=CoordinationDelegation(payload=request),
                    principal=admin_test_principal(),
                )
            assert (
                exc_info.value.failure_code
                is CoordinationFailureCode.AUTHORITY_SCOPE_MISMATCH
            )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="cross-task"))


@pytest.mark.asyncio
async def test_coordination_invokes_delegated_subtask_service() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    task_scope = mint_task_id()
    coordination_result = await _run_coordination(harness, task_scope=task_scope)
    assert coordination_result.delegated.delegation_id == "delegation-1"
    assert coordination_result.delegated.lease_id == TaskScopedAgentLeaseId(
        "lease-delegate-1",
    )


@pytest.mark.asyncio
async def test_coordination_child_execution_path_preserved() -> None:
    parent_execution_id: ExecutionId | None = None
    child_execution_id: ExecutionId | None = None
    child_parent_execution_id: ExecutionId | None = None
    child_run_id: RunId | None = None
    child_attempt_id: AttemptId | None = None

    class LineageDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            nonlocal child_execution_id, child_parent_execution_id
            nonlocal child_run_id, child_attempt_id
            child_execution_id = require_active_execution_id()
            child_parent_execution_id = peek_active_parent_execution_id()
            identity = peek_active_execution_identity()
            assert identity is not None
            child_run_id, child_attempt_id = identity
            return OcrResult(text=request.document_ref)

    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=LineageDelegate(),
    )
    root = _root_identity()
    parent_execution_id = root.execution_id
    task_scope = mint_task_id()
    coordination = _build_coordination_service(harness)
    harness.task_scope_authority.task_scope_id = task_scope

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await coordination.coordinate(
                _coordination_request(task_scope=task_scope),
                delegation=CoordinationDelegation(payload=request),
                principal=admin_test_principal(),
            )
            return result.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="lineage"))

    assert child_execution_id is not None
    assert parent_execution_id is not None
    assert child_execution_id != parent_execution_id
    assert child_parent_execution_id == parent_execution_id


@pytest.mark.asyncio
async def test_coordination_permission_scope_propagation_and_escalation_blocked() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    coordination = _build_coordination_service(harness)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(ChildExecutionFailedError) as exc_info:
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(
                        payload=request,
                        requested_permission_scopes=("read", "delete"),
                    ),
                    principal=admin_test_principal(),
                )
            from intergrax.contracts.delegation_authority import DelegationAuthorityError

            cause = exc_info.value.__cause__
            assert cause is not None
            assert isinstance(cause.__cause__, DelegationAuthorityError)
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.scoped(("read", "write")),
    ).execute(OcrRequest(document_ref="authority"))


@pytest.mark.asyncio
async def test_coordination_budget_propagation_and_escalation_blocked() -> None:
    from intergrax.runtime.execution.budget.models import (
        ExecutionBudgetReservationError,
    )

    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=50))
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    harness.service._child_execution = as_child_execution_port(
        ChildExecutionRunner[OcrRequest, OcrResult](ledger=ledger),
    )
    coordination = _build_coordination_service(harness)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=ledger,
            )
            try:
                with pytest.raises(ChildExecutionFailedError) as exc_info:
                    await coordination.coordinate(
                        _coordination_request(task_scope=task_scope),
                        delegation=CoordinationDelegation(
                            payload=request,
                            requested_budget=RunBudget(max_tool_calls=70),
                        ),
                        principal=admin_test_principal(),
                    )
            finally:
                reset_active_execution_budget(budget_token)
            cause = exc_info.value.__cause__
            assert cause is not None
            assert isinstance(cause.__cause__, ExecutionBudgetReservationError)
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="budget"))


class _FailingAcquisitionPort:
    def acquire(self, request, *, principal):
        del request, principal
        from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentError

        raise TaskScopedAgentError("acquisition failed")


@pytest.mark.asyncio
async def test_coordination_acquisition_failure_mapped() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    harness.service._task_scoped_agents._acquisition_service._acquisition = (
        _FailingAcquisitionPort()
    )
    coordination = _build_coordination_service(harness)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(AcquisitionFailedError):
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="acquire"))


class _FailingSpecialistDelegate:
    async def execute(self, request: OcrRequest) -> OcrResult:
        del request
        raise RuntimeError("specialist failed")


@pytest.mark.asyncio
async def test_coordination_child_failure_releases_lease() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_FailingSpecialistDelegate(),
    )
    coordination = _build_coordination_service(harness)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(ChildExecutionFailedError):
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="failed")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="child-fail"))
    lease = harness.lease_store.get(TaskScopedAgentLeaseId("lease-delegate-1"))
    assert lease is not None
    assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED


@pytest.mark.asyncio
async def test_coordination_lease_released_after_success() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    task_scope = mint_task_id()
    await _run_coordination(harness, task_scope=task_scope)
    lease = harness.lease_store.get(TaskScopedAgentLeaseId("lease-delegate-1"))
    assert lease is not None
    assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED


@dataclass
class _TrackingChildExecution:
    calls: int = 0

    async def execute_child(self, *, request, delegate, options=None):
        self.calls += 1
        return await delegate.execute(request)


@pytest.mark.asyncio
async def test_coordination_uses_child_execution_port_not_direct_runner() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    tracker = _TrackingChildExecution()
    harness.service._child_execution = tracker
    task_scope = mint_task_id()
    await _run_coordination(harness, task_scope=task_scope)
    assert tracker.calls == 1
