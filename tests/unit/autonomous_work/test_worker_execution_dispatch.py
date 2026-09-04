# © Artur Czarnecki. All rights reserved.

"""AW-5A — Worker execution dispatch tests."""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, TypeVar
from unittest.mock import AsyncMock

import pytest

from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionService,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryResponsibilityRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkerInstanceRepository,
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.principal_binding_resolver import WorkerPrincipalBindingResolver
from intergrax.autonomous_work.worker_execution_dispatch import WorkerExecutionDispatchService
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    PrincipalAuthorityGrantScopeKey,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.autonomous_work import (
    ResponsibilityStatus,
    WorkerExecutionDispatchDisposition,
    WorkerExecutionDispatchRejectionReason,
    WorkerExecutionDispatchRequest,
    WorkerExecutionSource,
    WorkerExecutionSourceKind,
    WorkerGoalStatus,
    WorkerLifecycleState,
    initial_revision,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    MembershipStatus,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.canonical_intake_adapter import CanonicalExecutionRuntimeAdapter
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.runtime import ExecutionRuntime
from intergrax.runtime.governance.root_execution_authority_admission import (
    DenyingRootExecutionAuthorityAdmission,
    RootExecutionAuthorityAdmissionService,
    UnavailableRootExecutionAuthorityAdmission,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = datetime(2026, 9, 4, 9, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-x"
_READ = "workspace.read"
_WRITE = "workspace.write"

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ProbePayload:
    value: str


@dataclass(frozen=True)
class ProbeResult:
    echoed: str


class RecordingExecutionIntake(Generic[PayloadT, ResultT]):
    """Spy intake port — records dispatch calls without owning authority."""

    def __init__(self, *, result_factory: object | None = None) -> None:
        self.calls: list[CanonicalExecutionIntakeRequest[PayloadT]] = []
        self._result_factory = result_factory

    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[PayloadT],
    ) -> CanonicalExecutionIntakeResult[ResultT]:
        self.calls.append(request)
        if self._result_factory is not None:
            return self._result_factory(request)
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        return CanonicalExecutionIntakeResult(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            result=ProbeResult(echoed="ok"),  # type: ignore[arg-type]
        )


def _binding_repo() -> InMemoryWorkerPrincipalBindingRepository:
    return InMemoryWorkerPrincipalBindingRepository()


def _seed_binding_and_authority(
    *,
    worker_id: str | None = None,
    principal_id: str = "principal-collaborative-1",
    authority_scopes: tuple[str, ...] = (_READ, _WRITE),
) -> tuple[
    str,
    InMemoryWorkerPrincipalBindingRepository,
    InMemoryWorkspaceMembershipRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryAuthorityDelegationRepository,
]:
    worker_id = worker_id or mint_worker_instance_id()
    binding_repo = _binding_repo()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=principal_id,
        )
    )
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id=f"membership-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo = InMemoryPrincipalAuthorityRepository()
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id=f"grant-{principal_id}",
            principal_id=principal_id,
            authority_scopes=authority_scopes,
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    delegation_repo = InMemoryAuthorityDelegationRepository()
    return worker_id, binding_repo, membership_repo, authority_repo, delegation_repo


def _dispatch_service(
    *,
    worker_repo: InMemoryWorkerInstanceRepository,
    binding_repo: InMemoryWorkerPrincipalBindingRepository,
    membership_repo: InMemoryWorkspaceMembershipRepository,
    authority_repo: InMemoryPrincipalAuthorityRepository,
    delegation_repo: InMemoryAuthorityDelegationRepository,
    intake: RecordingExecutionIntake[ProbePayload, ProbeResult] | None = None,
    root_admission: object | None = None,
    responsibility_repo: InMemoryResponsibilityRepository | None = None,
    goal_repo: InMemoryWorkerGoalRepository | None = None,
) -> tuple[WorkerExecutionDispatchService[ProbePayload, ProbeResult], RecordingExecutionIntake[ProbePayload, ProbeResult]]:
    recording = intake or RecordingExecutionIntake()
    service = WorkerExecutionDispatchService(
        worker_instance_repository=worker_repo,
        responsibility_repository=responsibility_repo or InMemoryResponsibilityRepository(),
        worker_goal_repository=goal_repo or InMemoryWorkerGoalRepository(),
        admission_service=WorkerExecutionAdmissionService(
            binding_resolver=WorkerPrincipalBindingResolver(binding_repo),
            authority_resolver=CollaborativeWorkAuthorityResolver(
                membership_repository=membership_repo,
                delegation_repository=delegation_repo,
                principal_authority_repository=authority_repo,
                clock=lambda: _UTC,
            ),
        ),
        root_authority_admission=root_admission or RootExecutionAuthorityAdmissionService(),
        execution_intake=recording,
    )
    return service, recording


def _active_worker(
    worker_repo: InMemoryWorkerInstanceRepository,
    *,
    worker_id: str,
) -> None:
    worker = contract_suite.worker_instance(
        worker_instance_id=worker_id,
        lifecycle_state=WorkerLifecycleState.ACTIVE,
        revision=initial_revision(),
    )
    worker_repo.create(worker)


def _dispatch_request(
    *,
    worker_id: str,
    source_ref: str = "operator/direct-1",
) -> WorkerExecutionDispatchRequest[ProbePayload, ProbeResult]:
    return WorkerExecutionDispatchRequest(
        worker_instance_id=worker_id,
        worker_revision=initial_revision(),
        requested_scopes=(_READ,),
        runtime_request=ExecutionRequest(
            input=ProbePayload(value="dispatch"),
            capabilities=frozenset({ExecutionCapability.AGENT}),
        ),
        source=WorkerExecutionSource(
            source_kind=WorkerExecutionSourceKind.OPERATOR,
            source_ref=source_ref,
        ),
        requested_at=_UTC,
    )


@pytest.mark.asyncio
async def test_happy_path_dispatches_through_canonical_intake() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    )

    result = await service.dispatch(_dispatch_request(worker_id=worker_id))

    assert result.disposition is WorkerExecutionDispatchDisposition.DISPATCHED
    assert result.correlation.run_id is not None
    assert result.correlation.attempt_id is not None
    assert result.correlation.execution_id is not None
    assert len(intake.calls) == 1
    assert intake.calls[0].tenant_id == _TENANT
    assert intake.calls[0].trusted_parent_execution_authority.permission_scopes == (_READ,)


@pytest.mark.asyncio
async def test_one_worker_many_executions() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    )

    first = await service.dispatch(_dispatch_request(worker_id=worker_id, source_ref="a"))
    second = await service.dispatch(_dispatch_request(worker_id=worker_id, source_ref="b"))

    assert first.correlation.execution_id != second.correlation.execution_id
    assert len(intake.calls) == 2


@pytest.mark.asyncio
async def test_same_source_ref_allows_multiple_executions() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    )

    first = await service.dispatch(_dispatch_request(worker_id=worker_id, source_ref="same"))
    second = await service.dispatch(_dispatch_request(worker_id=worker_id, source_ref="same"))

    assert first.correlation.execution_id != second.correlation.execution_id
    assert len(intake.calls) == 2


@pytest.mark.asyncio
async def test_collaborative_authority_denied_skips_runtime() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    )
    request = _dispatch_request(worker_id=worker_id)
    request = WorkerExecutionDispatchRequest(
        worker_instance_id=request.worker_instance_id,
        worker_revision=request.worker_revision,
        requested_scopes=("workspace.delete",),
        runtime_request=request.runtime_request,
        source=request.source,
        requested_at=request.requested_at,
    )

    result = await service.dispatch(request)

    assert result.disposition is WorkerExecutionDispatchDisposition.REJECTED
    assert result.rejection_reason is (
        WorkerExecutionDispatchRejectionReason.COLLABORATIVE_AUTHORITY_DENIED
    )
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_runtime_authority_denied_after_collaborative_allow() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        root_admission=DenyingRootExecutionAuthorityAdmission(),
    )

    result = await service.dispatch(_dispatch_request(worker_id=worker_id))

    assert result.disposition is WorkerExecutionDispatchDisposition.REJECTED
    assert result.rejection_reason is (
        WorkerExecutionDispatchRejectionReason.RUNTIME_AUTHORITY_DENIED
    )
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_runtime_unavailable_is_typed() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        root_admission=UnavailableRootExecutionAuthorityAdmission(),
    )

    result = await service.dispatch(_dispatch_request(worker_id=worker_id))

    assert result.disposition is WorkerExecutionDispatchDisposition.UNAVAILABLE
    assert len(intake.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lifecycle_state",
    [
        WorkerLifecycleState.STOPPED,
        WorkerLifecycleState.PAUSED,
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.PROVISIONING,
    ],
)
async def test_ineligible_worker_lifecycle_rejected(
    lifecycle_state: WorkerLifecycleState,
) -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    worker = contract_suite.worker_instance(
        worker_instance_id=worker_id,
        lifecycle_state=lifecycle_state,
        revision=initial_revision(),
    )
    worker_repo.create(worker)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    )

    result = await service.dispatch(_dispatch_request(worker_id=worker_id))

    assert result.disposition is WorkerExecutionDispatchDisposition.REJECTED
    assert result.rejection_reason is WorkerExecutionDispatchRejectionReason.WORKER_NOT_ELIGIBLE
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_stale_goal_revision_rejected() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    responsibility_repo = InMemoryResponsibilityRepository()
    goal_repo = InMemoryWorkerGoalRepository()
    responsibility = contract_suite.responsibility(worker_instance_id=worker_id)
    responsibility_repo.create(responsibility)
    goal = contract_suite.worker_goal(
        responsibility_id=responsibility.responsibility_id,
        status=WorkerGoalStatus.ACTIVE,
        revision=initial_revision(),
    )
    goal_repo.create(goal)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    request = WorkerExecutionDispatchRequest(
        worker_instance_id=worker_id,
        worker_revision=initial_revision(),
        requested_scopes=(_READ,),
        runtime_request=ExecutionRequest(
            input=ProbePayload(value="goal"),
            capabilities=frozenset({ExecutionCapability.AGENT}),
        ),
        source=WorkerExecutionSource(
            source_kind=WorkerExecutionSourceKind.GOAL_DECISION,
            source_ref="goal/decision/1",
        ),
        requested_at=_UTC,
        goal_id=goal.goal_id,
        goal_revision=initial_revision(),
        responsibility_id=responsibility.responsibility_id,
    )
    goal_repo.replace(goal, expected_revision=initial_revision())

    result = await service.dispatch(request)

    assert result.disposition is WorkerExecutionDispatchDisposition.REJECTED
    assert result.rejection_reason is WorkerExecutionDispatchRejectionReason.STALE_SOURCE
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_wrong_responsibility_owner_rejected() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    other_worker = mint_worker_instance_id()
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    responsibility_repo = InMemoryResponsibilityRepository()
    goal_repo = InMemoryWorkerGoalRepository()
    responsibility = contract_suite.responsibility(worker_instance_id=other_worker)
    responsibility_repo.create(responsibility)
    goal = contract_suite.worker_goal(
        responsibility_id=responsibility.responsibility_id,
        status=WorkerGoalStatus.ACTIVE,
        revision=initial_revision(),
    )
    goal_repo.create(goal)
    service, intake = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        responsibility_repo=responsibility_repo,
        goal_repo=goal_repo,
    )
    request = WorkerExecutionDispatchRequest(
        worker_instance_id=worker_id,
        worker_revision=initial_revision(),
        requested_scopes=(_READ,),
        runtime_request=ExecutionRequest(
            input=ProbePayload(value="goal"),
            capabilities=frozenset({ExecutionCapability.AGENT}),
        ),
        source=WorkerExecutionSource(
            source_kind=WorkerExecutionSourceKind.GOAL_DECISION,
            source_ref="goal/decision/2",
        ),
        requested_at=_UTC,
        goal_id=goal.goal_id,
        goal_revision=goal.revision,
        responsibility_id=responsibility.responsibility_id,
    )

    result = await service.dispatch(request)

    assert result.disposition is WorkerExecutionDispatchDisposition.REJECTED
    assert result.rejection_reason is WorkerExecutionDispatchRejectionReason.OWNERSHIP_MISMATCH
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_execution_runtime_invoked_exactly_once_per_dispatch() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)

    class ProbeDelegate:
        def __init__(self) -> None:
            self.execute = AsyncMock(return_value=ProbeResult(echoed="runtime"))

    delegate = ProbeDelegate()
    runtime = ExecutionRuntime(delegate)
    adapter = CanonicalExecutionRuntimeAdapter(runtime)
    service = WorkerExecutionDispatchService(
        worker_instance_repository=worker_repo,
        responsibility_repository=InMemoryResponsibilityRepository(),
        worker_goal_repository=InMemoryWorkerGoalRepository(),
        admission_service=WorkerExecutionAdmissionService(
            binding_resolver=WorkerPrincipalBindingResolver(binding_repo),
            authority_resolver=CollaborativeWorkAuthorityResolver(
                membership_repository=membership_repo,
                delegation_repository=delegation_repo,
                principal_authority_repository=authority_repo,
                clock=lambda: _UTC,
            ),
        ),
        root_authority_admission=RootExecutionAuthorityAdmissionService(),
        execution_intake=adapter,
    )

    result = await service.dispatch(_dispatch_request(worker_id=worker_id))

    delegate.execute.assert_awaited_once()
    assert result.disposition is WorkerExecutionDispatchDisposition.DISPATCHED
    assert result.correlation.execution_id is not None


@pytest.mark.asyncio
async def test_rejected_results_expose_no_runtime_ids() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    worker_repo = InMemoryWorkerInstanceRepository()
    _active_worker(worker_repo, worker_id=worker_id)
    service, _ = _dispatch_service(
        worker_repo=worker_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        root_admission=DenyingRootExecutionAuthorityAdmission(),
    )

    result = await service.dispatch(_dispatch_request(worker_id=worker_id))

    assert result.correlation.run_id is None
    assert result.correlation.attempt_id is None
    assert result.correlation.execution_id is None


def test_aw5a_dispatch_does_not_mint_parent_execution_authority() -> None:
    module = importlib.import_module("intergrax.autonomous_work.worker_execution_dispatch")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("delegation_authority"):
                for alias in node.names:
                    if alias.name == "ParentExecutionAuthority":
                        raise AssertionError(
                            "AW-5A dispatch must not import ParentExecutionAuthority"
                        )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if getattr(node.func.value, "id", None) == "ParentExecutionAuthority":
                raise AssertionError("AW-5A dispatch must not mint ParentExecutionAuthority")


def test_aw5a_dispatch_does_not_mint_execution_ids() -> None:
    module = importlib.import_module("intergrax.autonomous_work.worker_execution_dispatch")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    for token in ("mint_run_id", "mint_attempt_id", "mint_execution_id"):
        assert token not in source


def test_aw5a_contracts_do_not_define_worker_task_types() -> None:
    module = importlib.import_module("intergrax.contracts.autonomous_work.execution_dispatch")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("WorkerTask", "AutonomousTask", "WorkerRun", "WorkerAttempt"):
        assert forbidden not in source


def test_operator_path_works_without_workitem() -> None:
    request = _dispatch_request(worker_id=mint_worker_instance_id())
    assert request.collaborative_work_ref is None
    assert request.source.source_kind is WorkerExecutionSourceKind.OPERATOR


def test_admission_service_prepare_still_usable_for_aw3b() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority(authority_scopes=(_READ,))
    )
    admission = WorkerExecutionAdmissionService(
        binding_resolver=WorkerPrincipalBindingResolver(binding_repo),
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: _UTC,
        ),
    )
    context = admission.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_id,
            requested_authority_scopes=(_READ,),
        )
    )
    assert context.collaborative_authority_scopes == (_READ,)


def test_root_admission_mints_trusted_authority_in_governance_layer() -> None:
    from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
    from intergrax.contracts.delegation_authority import ParentExecutionAuthority
    from intergrax.contracts.runtime_execution_admission import (
        RootExecutionAuthorityAdmissionDisposition,
        RootExecutionAuthorityAdmissionRequest,
    )

    service = RootExecutionAuthorityAdmissionService()
    result = service.authorize(
        RootExecutionAuthorityAdmissionRequest(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id="principal-1",
            collaborative_authority_scopes=(_READ,),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=PolicyAction.ALLOW, reason="ok"),
            ),
        )
    )
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert isinstance(result.trusted_parent_execution_authority, ParentExecutionAuthority)
    assert result.trusted_parent_execution_authority.permission_scopes == (_READ,)
