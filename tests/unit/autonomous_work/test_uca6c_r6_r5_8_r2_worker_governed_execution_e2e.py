# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-R2 — Worker governed fulfillment execution E2E (qualified CodeCraft path)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.capability_acquisition_ports import (
    StaticWorkerCapabilityProfileResolver,
    permissive_capability_policy,
)
from intergrax.autonomous_work.host_available_capability_binding_service import (
    HostAvailableCapabilityBindingService,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.in_memory_worker_recovery_obstacle_capability_need_repository import (
    InMemoryWorkerRecoveryObstacleCapabilityNeedRepository,
)
from intergrax.autonomous_work.worker_capability_direct_reuse_fulfillment_service import (
    WorkerCapabilityDirectReuseFulfillmentService,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.autonomous_work.worker_recovery_governed_fulfillment_composition import (
    WorkerRecoveryGovernedFulfillmentWiring,
    build_worker_recovery_governed_fulfillment_wiring,
)
from intergrax.autonomous_work.worker_recovery_orchestration_service import (
    WorkerRecoveryOrchestrationService,
)
from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryOrchestrationDisposition,
)
from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingRequest,
    QualifiedCapabilityBindingResult,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.root_execution_launch import (
    RootExecutionLaunchPort,
    RootExecutionLaunchRequest,
    RootExecutionLaunchResult,
)
from intergrax.runtime.codecraft.artifact_reference import artifact_reference_for_craft
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CodeCraftQualifiedCapabilityBindingProvider,
)
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_dispatch_service import (
    QualifiedCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import (
    HumanApprovalResolutionError,
    HumanPauseCoordinator,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.active_task_registry_fulfillment_task_context_reader import (
    ActiveTaskRegistryFulfillmentTaskContextReader,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionOutcomeReader,
    CanonicalExecutionTerminalDisposition,
)
from intergrax.runtime.execution.execution_terminal_outcome_by_execution_id import (
    InMemoryExecutionTerminalOutcomeByExecutionIdStore,
    build_canonical_execution_outcome_reader,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_async_service import (
    WorkerRecoveryCapabilityFulfillmentAsyncService,
)
from intergrax.autonomous_work.in_memory_recovery_episode_repository import (
    InMemoryWorkerRecoveryEpisodeRepository,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryResponsibilityRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkerInstanceRepository,
)
from intergrax.autonomous_work.lifecycle import WorkerLifecycleService
from intergrax.autonomous_work.worker_capability_need_projection import (
    project_worker_capability_need_to_capability_need,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerDefinitionRepository,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
    _recovery_decision,
    _worker_need,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e import (
    _StaticDiscovery,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _PRINCIPAL,
    _WORKSPACE,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    RecordingHumanDecisionPort,
    RecordingRecoveryDispatchPort,
    _clock,
    _orchestration_request,
    _resume_target,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    _GOAL_ID,
    _RESP_ID,
    _WORKER_ID,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _RecordingExecutionPort,
)
from tests.unit.autonomous_work.uca6c_worker_authority_fixtures import (
    _READ,
    build_worker_execution_admission_for_uca6c,
)
from tests.unit.runtime.execution.test_uca6c_r6_r5_7_sequential_authority_generations import (
    _approve_current_pause,
    _build_handler,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clear_active_task_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


class _AllowingMsePort:
    def authorize(self, request, **kwargs):
        from intergrax.contracts.collaborative_work import (
            CollaborativeWorkEnforcementResult,
            PolicyCompositionResult,
        )
        from intergrax.contracts.meaningful_side_effect_authorization import (
            MeaningfulSideEffectAuthorizationResult,
        )
        from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="uca6c-r58-r2-allow-mse",
            policy_rule_id="test.uca6c.r58r2.mse",
        )
        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=True,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )


class _CountingRootLauncher(RootExecutionLaunchPort):
    def __init__(self, inner: RootExecutionLaunchPort) -> None:
        self._inner = inner
        self.launch_count = 0

    async def launch(
        self, request: RootExecutionLaunchRequest
    ) -> RootExecutionLaunchResult:
        self.launch_count += 1
        return await self._inner.launch(request)


@dataclass
class _CraftAlignedAcquisitionStrategy:
    strategy_id: str
    artifact_reference: str
    calls: int = 0

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        return frozenset({CapabilityKind.TOOL})

    def supports(self, request: CapabilityAcquisitionRequest) -> bool:
        return True

    def acquire(
        self, request: CapabilityAcquisitionRequest
    ) -> CapabilityAcquisitionResult:
        self.calls += 1
        return CapabilityAcquisitionResult(
            request_id=request.request_id,
            gap_id=request.capability_gap.gap_id,
            strategy_id=self.strategy_id,
            outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
            reason_code=CapabilityAcquisitionReasonCode.NONE,
            started_at=request.requested_at,
            completed_at=request.requested_at,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
            evidence=CapabilityAcquisitionEvidence(
                artifact_reference=self.artifact_reference,
            ),
        )


@dataclass
class _ArtifactQualification:
    artifact: str
    calls: int = 0

    def qualify(self, request):
        self.calls += 1
        return CapabilityQualificationResult(
            qualification_request_id=request.qualification_request_id,
            acquisition_request_id=request.acquisition_request_id,
            gap_id=request.gap_id,
            strategy_id=request.strategy_id,
            provider_id="codecraft.qualification",
            outcome=CapabilityQualificationOutcome.QUALIFIED,
            reason_code=CapabilityQualificationReasonCode.NONE,
            started_at=request.requested_at,
            completed_at=request.requested_at,
            evidence=CapabilityQualificationEvidence(
                provider_id="codecraft.qualification",
                qualification_request_id=request.qualification_request_id,
                acquisition_request_id=request.acquisition_request_id,
                acquisition_strategy_id=request.strategy_id,
                gap_id=request.gap_id,
                artifact_reference=self.artifact,
            ),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


@dataclass
class _CountingCodeCraftBindingProvider:
    inner: CodeCraftQualifiedCapabilityBindingProvider
    bind_calls: int = 0

    @property
    def provider_id(self) -> str:
        return self.inner.provider_id

    def supports(self, request: QualifiedCapabilityBindingRequest) -> bool:
        return self.inner.supports(request)

    def bind(
        self, request: QualifiedCapabilityBindingRequest
    ) -> QualifiedCapabilityBindingResult:
        self.bind_calls += 1
        return self.inner.bind(request)


@dataclass
class _CountingAsyncRecoveryFulfillment:
    inner: WorkerRecoveryCapabilityFulfillmentAsyncService
    calls: int = 0

    async def fulfill_recovery_capability_async(self, handoff):
        self.calls += 1
        return await self.inner.fulfill_recovery_capability_async(handoff)


@dataclass(frozen=True)
class _GovernedE2EStack:
    wiring: WorkerRecoveryGovernedFulfillmentWiring
    root_launcher: _CountingRootLauncher
    acquisition: _CraftAlignedAcquisitionStrategy
    qualification: _ArtifactQualification
    discovery: _StaticDiscovery
    binding: _CountingCodeCraftBindingProvider
    async_fulfillment: _CountingAsyncRecoveryFulfillment
    handler: object
    composition: object
    hitl: object
    checkpoint_store: object
    backend: object
    craft_id: str
    task: Task
    run_id: RunId
    attempt_id: AttemptId
    terminal_outcome_store: InMemoryExecutionTerminalOutcomeByExecutionIdStore


def _aligned_worker_need():
    return replace(
        _worker_need(),
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:uca6b-1",
    )


def _missing_capability_discovery() -> _StaticDiscovery:
    need = _aligned_worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-r58-r2-governed",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    return _StaticDiscovery(completion)


def _seed_obstacle_need_repo(
    need_repo: InMemoryWorkerRecoveryObstacleCapabilityNeedRepository,
    principal_repo: InMemoryWorkerPrincipalBindingRepository,
) -> None:
    need = _aligned_worker_need()
    need_repo.record_obstacle_capability_need(need)
    principal_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=_WORKER_ID,
            tenant_id=_TENANT,
        ),
    )


def _build_governed_e2e_stack(
    tmp_path: Path,
    *,
    tool_wiring_context,
    craft_id: str,
    handler,
    terminal_outcome_store: InMemoryExecutionTerminalOutcomeByExecutionIdStore,
) -> _GovernedE2EStack:
    artifact = artifact_reference_for_craft(craft_id)
    dispatch, delegate, inner_launcher = (
        build_qualified_capability_execution_dispatch_service(
            handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry(
                (handler,)
            ),
            runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
            terminal_outcome_store=terminal_outcome_store,
        )
    )
    root_launcher = _CountingRootLauncher(inner_launcher)
    dispatch = QualifiedCapabilityExecutionDispatchService(
        root_execution_launcher=root_launcher,
        runtime_delegate=delegate,
    )
    binding_inner = CodeCraftQualifiedCapabilityBindingProvider(tool_wiring_context)
    binding_provider = _CountingCodeCraftBindingProvider(inner=binding_inner)
    binding = QualifiedCapabilityBindingService((binding_provider,))
    discovery = _missing_capability_discovery()
    strategy = _CraftAlignedAcquisitionStrategy(
        strategy_id="codecraft.synthesis.v1",
        artifact_reference=artifact,
    )
    acquisition = CapabilityAcquisitionService(
        strategies=(strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    qualification = _ArtifactQualification(artifact=artifact)
    recovery = WorkerCapabilityRecoveryCoordinator(
        discovery=discovery,
        acquisition=acquisition,
        qualification=qualification,
    )
    authority_admission = build_worker_execution_admission_for_uca6c(
        worker_instance_id=_WORKER_ID,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    direct_reuse = WorkerCapabilityDirectReuseFulfillmentService(
        binding=HostAvailableCapabilityBindingService(()),
        execution=_RecordingExecutionPort(),
        authority_admission=authority_admission,
    )
    need_repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    principal_repo = InMemoryWorkerPrincipalBindingRepository()
    _seed_obstacle_need_repo(need_repo, principal_repo)
    profile_resolver = StaticWorkerCapabilityProfileResolver(
        permissive_capability_policy(_PROFILE),
    )
    wiring = build_worker_recovery_governed_fulfillment_wiring(
        recovery=recovery,
        direct_reuse=direct_reuse,
        inner_dispatch=dispatch,
        binding=binding,
        obstacle_capability_need_reader=need_repo,
        task_context_reader=ActiveTaskRegistryFulfillmentTaskContextReader(),
        principal_binding_repository=principal_repo,
        capability_profile_resolver=profile_resolver,
        authority_admission=authority_admission,
    )
    async_fulfillment = _CountingAsyncRecoveryFulfillment(
        inner=wiring.fulfillment_async,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(tenant_id=_TENANT, user_id="u1", message="r58-r2", task_id=_TASK_ID)
    return _GovernedE2EStack(
        wiring=wiring,
        root_launcher=root_launcher,
        acquisition=strategy,
        qualification=qualification,
        discovery=discovery,
        binding=binding_provider,
        async_fulfillment=async_fulfillment,
        handler=handler,
        composition=None,
        hitl=None,
        checkpoint_store=None,
        backend=None,
        craft_id=craft_id,
        task=task,
        run_id=run_id,
        attempt_id=attempt_id,
        terminal_outcome_store=terminal_outcome_store,
    )


def _build_orchestration_service(
    stack: _GovernedE2EStack,
    *,
    outcome_reader: CanonicalExecutionOutcomeReader,
) -> tuple[WorkerRecoveryOrchestrationService, dict[str, object]]:
    worker_repo = InMemoryWorkerInstanceRepository()
    goal_repo = InMemoryWorkerGoalRepository()
    responsibility_repo = InMemoryResponsibilityRepository()
    continuity_repo = InMemoryWorkContinuityStateRepository()
    episode_repo = InMemoryWorkerRecoveryEpisodeRepository()
    definition_repo = InMemoryWorkerDefinitionRepository()
    definition = contract_suite.worker_definition()
    definition_repo.create(definition)
    worker = contract_suite.worker_instance(
        worker_instance_id=_WORKER_ID,
        worker_definition_id=definition.worker_definition_id,
        lifecycle_state=WorkerLifecycleState.WORKING,
    )
    worker_repo.create(worker)
    responsibility = contract_suite.responsibility(
        responsibility_id=_RESP_ID,
        worker_instance_id=_WORKER_ID,
    )
    responsibility_repo.create(responsibility)
    goal = contract_suite.worker_goal(
        goal_id=_GOAL_ID,
        responsibility_id=_RESP_ID,
    )
    goal_repo.create(goal)
    continuity_repo.create(
        contract_suite.continuity_state(worker_instance_ref=_WORKER_ID)
    )
    dispatch = RecordingRecoveryDispatchPort()
    human = RecordingHumanDecisionPort()
    service = WorkerRecoveryOrchestrationService(
        episode_repository=episode_repo,
        worker_instance_repository=worker_repo,
        worker_goal_repository=goal_repo,
        continuity_repository=continuity_repo,
        lifecycle_service=WorkerLifecycleService(repository=worker_repo, clock=_clock),
        dispatch_port=dispatch,
        human_decision_port=human,
        recovery_capability_fulfillment_port=stack.wiring.fulfillment_sync,
        recovery_capability_fulfillment_async_port=stack.async_fulfillment,
        recovery_capability_fulfillment_request_builder=stack.wiring.request_builder,
        execution_outcome_reader=outcome_reader,
        clock=_clock,
    )
    return service, {
        "dispatch": dispatch,
        "episode_repo": episode_repo,
        "worker_repo": worker_repo,
    }


def _orch_request_for_stack(stack: _GovernedE2EStack):
    need = _aligned_worker_need()
    decision = _recovery_decision(need)
    return _orchestration_request(
        decision=decision,
        resume_target=_resume_target(
            run_id=stack.run_id,
            requested_scopes=(_READ,),
        ),
    )


async def _register_task(stack: _GovernedE2EStack) -> None:
    await ActiveTaskRegistry.register(stack.task, stack.run_id)


def _continuation_for_worker_pause(
    task: Task,
    composition,
) -> tuple[str, object]:
    human_request = task.runtime.governance.human_request
    assert human_request is not None
    governed = human_request.governed_continuation
    assert governed is not None
    continuation_id = governed.continuation_request_id
    reentry = composition.suspended_work_reentry_coordinator
    assert reentry is not None
    descriptor = reentry.store.load_active_for_continuation(continuation_id)
    assert descriptor is not None
    return continuation_id, descriptor


def _approve_until_backend(
    stack: _GovernedE2EStack,
    *,
    composition,
    hitl,
    checkpoint_store,
    backend,
    execution_id: ExecutionId,
    max_rounds: int = 6,
) -> None:
    for _ in range(max_rounds):
        if backend.calls >= 1:
            return
        if stack.task.state is not TaskState.WAITING_FOR_HUMAN:
            return
        pause_record = stack.task.runtime.governance.pause_record
        human_request = stack.task.runtime.governance.human_request
        if pause_record is None or human_request is None:
            return
        continuation_id, descriptor = _continuation_for_worker_pause(
            stack.task,
            composition,
        )
        id_token = bind_active_execution_identity(
            run_id=descriptor.identity.run_id,
            attempt_id=descriptor.identity.attempt_id,
            execution_id=execution_id,
        )
        try:
            _approve_current_pause(
                stack.task,
                hitl=hitl,
                continuation_id=continuation_id,
                run_id=descriptor.identity.run_id,
                attempt_id=descriptor.identity.attempt_id,
                execution_id=execution_id,
                checkpoint_store=checkpoint_store,
            )
        finally:
            reset_active_execution_identity(id_token)
    assert backend.calls == 1


def _build_full_handler_stack(
    tmp_path: Path,
    *,
    terminal_outcome_store: InMemoryExecutionTerminalOutcomeByExecutionIdStore,
):
    (
        handler,
        composition,
        _,
        craft_id,
        hitl,
        checkpoint_store,
        backend,
        _,
        codecraft_composition,
    ) = _build_handler(
        tmp_path,
        _AllowingMsePort(),
        terminal_outcome_store=terminal_outcome_store,
    )
    tool_ctx = codecraft_composition.tool_wiring_context
    return handler, composition, craft_id, hitl, checkpoint_store, backend, tool_ctx


@pytest.mark.asyncio
async def test_worker_governed_execution_pause_resume_single_backend(
    tmp_path: Path,
) -> None:
    terminal_outcome_store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    handler, composition, craft_id, hitl, checkpoint_store, backend, tool_ctx = (
        _build_full_handler_stack(
            tmp_path,
            terminal_outcome_store=terminal_outcome_store,
        )
    )
    stack = _build_governed_e2e_stack(
        tmp_path,
        tool_wiring_context=tool_ctx,
        craft_id=craft_id,
        handler=handler,
        terminal_outcome_store=terminal_outcome_store,
    )
    stack = replace(
        stack,
        composition=composition,
        hitl=hitl,
        checkpoint_store=checkpoint_store,
        backend=backend,
    )
    await _register_task(stack)
    outcome_reader = build_canonical_execution_outcome_reader(terminal_outcome_store)
    service, ctx = _build_orchestration_service(stack, outcome_reader=outcome_reader)
    orch_request = _orch_request_for_stack(stack)
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        ),
    )
    try:
        first = await service.orchestrate(orch_request)
        assert (
            first.disposition
            is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED
        ), first.episode.terminal_reason
        execution_id = first.episode.last_execution_id
        assert execution_id is not None
        assert stack.root_launcher.launch_count == 1
        assert stack.async_fulfillment.calls == 1
        assert stack.discovery.calls == 1
        assert stack.acquisition.calls == 1
        assert stack.qualification.calls == 1
        assert stack.binding.bind_calls == 1
        assert backend.calls == 0
        worker = ctx["worker_repo"].get(
            worker_instance_id=first.episode.worker_instance_id
        )
        assert worker is not None
        assert worker.lifecycle_state is WorkerLifecycleState.WAITING_EXTERNAL

        pause_record = stack.task.runtime.governance.pause_record
        human_request = stack.task.runtime.governance.human_request
        assert pause_record is not None and human_request is not None
        assert human_request.governed_continuation is not None
        reentry = composition.suspended_work_reentry_coordinator
        assert reentry is not None
        _approve_until_backend(
            stack,
            composition=composition,
            hitl=hitl,
            checkpoint_store=checkpoint_store,
            backend=backend,
            execution_id=execution_id,
        )
        assert stack.root_launcher.launch_count == 1
        assert stack.acquisition.calls == 1
        assert stack.qualification.calls == 1
        assert stack.binding.bind_calls == 1
        terminal = outcome_reader.get_terminal_outcome(execution_id)
        assert (
            terminal.disposition is CanonicalExecutionTerminalDisposition.SUCCEEDED
        )

        final = await service.orchestrate(orch_request)
        assert final.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
        assert stack.async_fulfillment.calls == 1
        assert stack.root_launcher.launch_count == 1
        assert backend.calls == 1
        assert final.episode.last_execution_id == execution_id
    finally:
        reset_active_execution_governance_identity(gov_token)


@pytest.mark.asyncio
async def test_worker_governed_no_reacquisition_after_resume(
    tmp_path: Path,
) -> None:
    terminal_outcome_store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    handler, composition, craft_id, hitl, checkpoint_store, backend, tool_ctx = (
        _build_full_handler_stack(
            tmp_path,
            terminal_outcome_store=terminal_outcome_store,
        )
    )
    stack = _build_governed_e2e_stack(
        tmp_path,
        tool_wiring_context=tool_ctx,
        craft_id=craft_id,
        handler=handler,
        terminal_outcome_store=terminal_outcome_store,
    )
    await _register_task(stack)
    outcome_reader = build_canonical_execution_outcome_reader(terminal_outcome_store)
    service, _ = _build_orchestration_service(stack, outcome_reader=outcome_reader)
    orch_request = _orch_request_for_stack(stack)
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        ),
    )
    try:
        first = await service.orchestrate(orch_request)
        acq_before = stack.acquisition.calls
        qual_before = stack.qualification.calls
        bind_before = stack.binding.bind_calls
        disc_before = stack.discovery.calls
        execution_id = first.episode.last_execution_id
        assert execution_id is not None
        continuation_id, descriptor = _continuation_for_worker_pause(
            stack.task,
            composition,
        )
        id_token = bind_active_execution_identity(
            run_id=descriptor.identity.run_id,
            attempt_id=descriptor.identity.attempt_id,
            execution_id=execution_id,
        )
        try:
            _approve_current_pause(
                stack.task,
                hitl=hitl,
                continuation_id=continuation_id,
                run_id=descriptor.identity.run_id,
                attempt_id=descriptor.identity.attempt_id,
                execution_id=execution_id,
                checkpoint_store=checkpoint_store,
            )
        finally:
            reset_active_execution_identity(id_token)
        assert stack.acquisition.calls == acq_before
        assert stack.qualification.calls == qual_before
        assert stack.binding.bind_calls == bind_before
        assert stack.discovery.calls == disc_before
    finally:
        reset_active_execution_governance_identity(gov_token)


@pytest.mark.asyncio
async def test_worker_governed_stale_approval_blocks_backend(
    tmp_path: Path,
) -> None:
    terminal_outcome_store = InMemoryExecutionTerminalOutcomeByExecutionIdStore()
    handler, composition, craft_id, hitl, checkpoint_store, backend, tool_ctx = (
        _build_full_handler_stack(
            tmp_path,
            terminal_outcome_store=terminal_outcome_store,
        )
    )
    stack = _build_governed_e2e_stack(
        tmp_path,
        tool_wiring_context=tool_ctx,
        craft_id=craft_id,
        handler=handler,
        terminal_outcome_store=terminal_outcome_store,
    )
    await _register_task(stack)
    service, _ = _build_orchestration_service(
        stack,
        outcome_reader=build_canonical_execution_outcome_reader(terminal_outcome_store),
    )
    orch_request = _orch_request_for_stack(stack)
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        ),
    )
    try:
        first = await service.orchestrate(orch_request)
        pause_record = stack.task.runtime.governance.pause_record
        human_request = stack.task.runtime.governance.human_request
        assert pause_record is not None and human_request is not None
        assert first.episode.last_execution_id is not None
        continuation_id, descriptor = _continuation_for_worker_pause(
            stack.task,
            composition,
        )
        execution_id = descriptor.identity.execution_id
        id_token = bind_active_execution_identity(
            run_id=descriptor.identity.run_id,
            attempt_id=descriptor.identity.attempt_id,
            execution_id=execution_id,
        )
        try:
            with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
                HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
                    stack.task,
                    HumanResponseVerdict.APPROVE,
                    approver=local_development_approver_evidence(
                        tenant_id=stack.task.tenant_id,
                    ),
                    continuation=hitl.port,
                    pause_id="stale_pause_id",
                    human_request_id=human_request.request_id,
                    run_id=str(descriptor.identity.run_id),
                    attempt_id=str(descriptor.identity.attempt_id),
                    execution_id=str(execution_id),
                )
        finally:
            reset_active_execution_identity(id_token)
        assert backend.calls == 0
        assert stack.root_launcher.launch_count == 1
        assert stack.acquisition.calls == 1
    finally:
        reset_active_execution_governance_identity(gov_token)
