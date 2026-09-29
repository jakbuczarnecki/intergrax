# © Artur Czarnecki. All rights reserved.

"""AW-7C-P4 qualification + governance + canonical execution composition."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Generic, TypeVar

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
from intergrax.autonomous_work.scoped_adaptive_integration import (
    WorkerScopedAdaptiveIntegrationOrchestrationService,
)
from intergrax.autonomous_work.scoped_adaptive_integration_execution import (
    WorkerScopedAdaptiveIntegrationExecutionCoordinator,
)
from intergrax.autonomous_work.worker_execution_dispatch import WorkerExecutionDispatchService
from intergrax.capability_qualification.qualification_service import (
    CapabilityQualificationService,
)
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
    WorkerLifecycleState,
    initial_revision,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityProfileRef,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionSource,
    WorkerExecutionSourceKind,
)
from intergrax.contracts.autonomous_work.profile_reference import initial_profile_version
from intergrax.contracts.autonomous_work.scoped_adaptive_integration import (
    ScopedAdaptiveIntegrationCorrelation,
    ScopedAdaptiveIntegrationExecutionRequest,
)
from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionOutcome,
    ScopedAdaptiveIntegrationExecutionOutput,
    ScopedAdaptiveIntegrationExecutionRequest as P4ExecutionRequest,
    ScopedAdaptiveIntegrationWorkerDispatchContext,
)
from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionHandoff,
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
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RootExecutionAdmissionPolicyRule,
    WORKER_ROOT_EXECUTION_OPERATION,
)
from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationScope,
)
from intergrax.integrations.qualification.reference_scoped_integration_adaptation import (
    REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
    ReferenceScopedIntegrationAdaptationStrategy,
    reference_read_operation,
    reference_write_operation,
)
from intergrax.integrations.qualification.reference_scoped_integration_adaptation_target_source import (
    reference_scoped_integration_adaptation_target_source,
)
from intergrax.integrations.qualification.reference_scoped_integration_qualification import (
    ReferenceScopedIntegrationQualificationProvider,
)
from intergrax.integrations.scoped_integration_adaptation_service import (
    ScopedIntegrationAdaptationPortAdapter,
)
from intergrax.integrations.scoped_integration_adaptation_target_resolver import (
    SourceBackedScopedIntegrationAdaptationTargetResolver,
)
from intergrax.runtime.governance.default_root_execution_launcher import (
    DefaultRootExecutionLauncher,
)
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission_from_rules,
)
from intergrax.runtime.governance.root_execution_authority_admission import (
    DenyingRootExecutionAuthorityAdmission,
    UnavailableRootExecutionAuthorityAdmission,
)
from intergrax.contracts.runtime_policy import PolicyAction
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_TS = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-x"
_READ = "workspace.read"
_WRITE = "workspace.write"
_WORKER_ID = contract_suite.mint_worker_instance_id()
_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)

HandoffT = TypeVar("HandoffT")
ResultT = TypeVar("ResultT")


class _RecordingIntake(Generic[HandoffT, ResultT]):
    def __init__(self) -> None:
        self.calls: list[CanonicalExecutionIntakeRequest[HandoffT]] = []

    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[HandoffT],
    ) -> CanonicalExecutionIntakeResult[ResultT]:
        self.calls.append(request)
        return CanonicalExecutionIntakeResult(
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            result=ScopedAdaptiveIntegrationExecutionOutput(
                evidence_ref="exec-evidence-1",
                tenant_id=_TENANT,
            ),  # type: ignore[arg-type]
        )


def _decision() -> WorkerCapabilityAcquisitionDecision:
    candidate = WorkerCapabilityCandidate(
        candidate_id="cand-1",
        candidate_kind=WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION,
        capability_ref="integration:demo",
        source_domain="integrations",
        operations=("READ_CONFIGURATION", "WRITE_CONFIGURATION"),
        risk_class=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
        evidence_refs=(),
        discovered_at=_TS,
    )
    return WorkerCapabilityAcquisitionDecision(
        decision_id="dec-1",
        worker_instance_id=_WORKER_ID,
        obstacle_id="obs-1",
        recovery_decision_id="rec-1",
        need_id="need-1",
        disposition=CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE,
        capability_profile_ref=_PROFILE,
        reason_code=CapabilityAcquisitionReasonCode.A2_ADAPTATION_REQUIRED,
        evidence_refs=(),
        decided_at=_TS,
        selected_candidate=candidate,
        autonomy_level=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
    )


def _scope() -> ScopedIntegrationAdaptationScope:
    return ScopedIntegrationAdaptationScope(
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.MESSAGE_BUS,
        provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
        resource_scope="rs-1",
        permitted_operations=(reference_read_operation(), reference_write_operation()),
        network_allowlist=NetworkEgressAllowlist(
            hosts=(
                NetworkEgressHost(scheme="https", hostname="a.example.com", port=443),
            ),
        ),
        credential_grant_ref="grant-1",
        expires_at=_TS + timedelta(hours=1),
        candidate_id="cand-1",
        candidate_revision="rev-1",
    )


def _prepare():
    decision = _decision()
    candidate = decision.selected_candidate
    assert candidate is not None
    orchestration = WorkerScopedAdaptiveIntegrationOrchestrationService(
        adaptation_port=ScopedIntegrationAdaptationPortAdapter(
            target_resolver=SourceBackedScopedIntegrationAdaptationTargetResolver(
                target_source=reference_scoped_integration_adaptation_target_source(),
            ),
            strategies=(ReferenceScopedIntegrationAdaptationStrategy(),),
        ),
    )
    request = ScopedAdaptiveIntegrationExecutionRequest(
        worker_instance_id=_WORKER_ID,
        correlation=ScopedAdaptiveIntegrationCorrelation(tenant_id=_TENANT),
        acquisition_decision=decision,
        selected_candidate=candidate,
        need_id="need-1",
        recovery_decision_id="rec-1",
        integration_capability_ref="integration:demo",
        required_operations=("READ_CONFIGURATION", "WRITE_CONFIGURATION"),
        adaptation_scope=_scope(),
        requested_at=_TS,
    )
    return orchestration.prepare(request)


def _seed_worker(binding_repo: InMemoryWorkerPrincipalBindingRepository) -> None:
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=_WORKER_ID,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id="principal-1",
        )
    )


def _dispatch_stack(
    intake: _RecordingIntake[ScopedAdaptiveIntegrationExecutionHandoff, ScopedAdaptiveIntegrationExecutionOutput],
    *,
    root_admission: object | None = None,
) -> WorkerExecutionDispatchService[
    ScopedAdaptiveIntegrationExecutionHandoff,
    ScopedAdaptiveIntegrationExecutionOutput,
]:
    worker_repo = InMemoryWorkerInstanceRepository()
    worker_repo.create(
        contract_suite.worker_instance(
            worker_instance_id=_WORKER_ID,
            lifecycle_state=WorkerLifecycleState.ACTIVE,
            revision=initial_revision(),
        )
    )
    binding_repo = InMemoryWorkerPrincipalBindingRepository()
    _seed_worker(binding_repo)
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id="principal-1",
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo = InMemoryPrincipalAuthorityRepository()
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-1",
            principal_id="principal-1",
            authority_scopes=(_READ, _WRITE),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    delegation_repo = InMemoryAuthorityDelegationRepository()
    if root_admission is None:
        root_admission = build_root_execution_authority_admission_from_rules(
            root_execution_admission_rules=(
                RootExecutionAdmissionPolicyRule(
                    rule_id="test.worker.root_execution.allow",
                    decision=PolicyAction.ALLOW,
                    execution_operation=WORKER_ROOT_EXECUTION_OPERATION,
                ),
            ),
        )
    return WorkerExecutionDispatchService(
        worker_instance_repository=worker_repo,
        responsibility_repository=InMemoryResponsibilityRepository(),
        worker_goal_repository=InMemoryWorkerGoalRepository(),
        admission_service=WorkerExecutionAdmissionService(
            binding_resolver=WorkerPrincipalBindingResolver(binding_repo),
            authority_resolver=CollaborativeWorkAuthorityResolver(
                membership_repository=membership_repo,
                delegation_repository=delegation_repo,
                principal_authority_repository=authority_repo,
                clock=lambda: _TS,
            ),
        ),
        root_execution_launcher=DefaultRootExecutionLauncher(
            root_authority_admission=root_admission,
            execution_intake=intake,
        ),
    )


def _p4_request(
    preparation,
    *,
    idempotency_key: str = "idem-1",
) -> P4ExecutionRequest:
    return P4ExecutionRequest(
        preparation=preparation,
        worker_dispatch=ScopedAdaptiveIntegrationWorkerDispatchContext(
            worker_revision=initial_revision(),
            requested_scopes=(_READ, _WRITE),
            source=WorkerExecutionSource(
                source_kind=WorkerExecutionSourceKind.RECOVERY,
                source_ref="recovery-1",
            ),
        ),
        tenant_id=_TENANT,
        execution_idempotency_key=idempotency_key,
        requested_at=_TS,
    )


@pytest.mark.asyncio
async def test_p4_success_dispatches_canonical_intake_once() -> None:
    preparation = _prepare()
    intake = _RecordingIntake[
        ScopedAdaptiveIntegrationExecutionHandoff,
        ScopedAdaptiveIntegrationExecutionOutput,
    ]()
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert result.run_id is not None
    assert result.attempt_id is not None
    assert result.execution_id is not None
    assert len(intake.calls) == 1


@pytest.mark.asyncio
async def test_p4_governance_denied_zero_intake() -> None:
    preparation = _prepare()
    intake = _RecordingIntake()
    dispatch = _dispatch_stack(intake, root_admission=DenyingRootExecutionAuthorityAdmission())
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.GOVERNANCE_DENIED
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_p4_governance_unavailable_zero_intake() -> None:
    preparation = _prepare()
    intake = _RecordingIntake()
    dispatch = _dispatch_stack(
        intake,
        root_admission=UnavailableRootExecutionAuthorityAdmission(),
    )
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.GOVERNANCE_UNAVAILABLE
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_p4_no_qualification_provider_zero_intake() -> None:
    preparation = _prepare()
    intake = _RecordingIntake()
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(()),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED
    assert len(intake.calls) == 0


@pytest.mark.asyncio
async def test_p4_duplicate_idempotency_key_fails_closed() -> None:
    preparation = _prepare()
    intake = _RecordingIntake()
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    first = await coordinator.execute(_p4_request(preparation, idempotency_key="dup-key"))
    assert first.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    second = await coordinator.execute(_p4_request(preparation, idempotency_key="dup-key"))
    assert second.outcome is ScopedAdaptiveIntegrationExecutionOutcome.DUPLICATE_INVOCATION
    assert len(intake.calls) == 1


@pytest.mark.asyncio
async def test_p4_tenant_mismatch_rejected_before_dispatch() -> None:
    preparation = _prepare()
    intake = _RecordingIntake()
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    bad = _p4_request(preparation)
    bad = P4ExecutionRequest(
        preparation=bad.preparation,
        worker_dispatch=bad.worker_dispatch,
        tenant_id="tenant-b",
        execution_idempotency_key="idem-tenant",
        requested_at=bad.requested_at,
    )
    result = await coordinator.execute(bad)
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED
    assert len(intake.calls) == 0
