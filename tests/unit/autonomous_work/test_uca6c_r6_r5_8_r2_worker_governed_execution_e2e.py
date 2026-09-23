# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-R2 — Worker governed fulfillment execution E2E (qualified CodeCraft path)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.host_available_capability_binding_service import (
    HostAvailableCapabilityBindingService,
)
from intergrax.autonomous_work.worker_capability_direct_reuse_fulfillment_service import (
    WorkerCapabilityDirectReuseFulfillmentService,
)
from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_service import (
    WorkerRecoveryCapabilityFulfillmentService,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryOrchestrationDisposition,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
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
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
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
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.task.task import Task
from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionTerminalDisposition,
)
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _worker_need,
)
from intergrax.autonomous_work.worker_capability_need_projection import (
    project_worker_capability_need_to_capability_need,
)
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e import (
    _StaticDiscovery,
    _fulfillment_request,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _PRINCIPAL,
    _WORKSPACE,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import _WORKER_ID
from tests.unit.autonomous_work.uca6c_worker_authority_fixtures import (
    build_worker_execution_admission_for_uca6c,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    StubExecutionOutcomeReader,
    _decision,
    _harness,
    _orchestration_request,
)
from intergrax.capability_acquisition.acquisition_service import CapabilityAcquisitionService
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
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
from tests.unit.runtime.execution.test_uca6c_r6_r5_7_sequential_authority_generations import (
    _approve_current_pause,
    _build_handler,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)


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

    async def launch(self, request: RootExecutionLaunchRequest) -> RootExecutionLaunchResult:
        self.launch_count += 1
        return await self._inner.launch(request)


class _GovernedTaskExecutionAdapter:
    def __init__(
        self,
        inner: WorkerQualifiedCapabilityExecutionEngineAdapter,
        task: Task,
    ) -> None:
        self._inner = inner
        self._task = task

    def execute(self, request):
        token = bind_governed_execution_task(self._task)
        try:
            return self._inner.execute(request)
        finally:
            reset_governed_execution_task(token)


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

    def acquire(self, request: CapabilityAcquisitionRequest) -> CapabilityAcquisitionResult:
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
class _RecordingFulfillmentCoordinator:
    inner: WorkerCapabilityFulfillmentCoordinator
    calls: int = 0

    def fulfill(self, request: WorkerCapabilityFulfillmentRequest, *, decided_at=None):
        self.calls += 1
        return self.inner.fulfill(request, decided_at=decided_at)


@dataclass
class _StaticFulfillmentRequestBuilder:
    fulfillment_request: WorkerCapabilityFulfillmentRequest

    def build_fulfillment_request(self, *, episode, request):
        return self.fulfillment_request


def _missing_capability_discovery() -> _StaticDiscovery:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-r58-r2-governed",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    return _StaticDiscovery(completion)


def _build_worker_governed_stack(
    tmp_path: Path,
    *,
    task: Task,
    tool_wiring_context,
    craft_id: str,
    handler,
) -> tuple[
    _RecordingFulfillmentCoordinator,
    _CountingRootLauncher,
    object,
    object,
    object,
    object,
]:
    artifact = artifact_reference_for_craft(craft_id)
    dispatch, delegate, inner_launcher = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    counting_launcher = _CountingRootLauncher(inner_launcher)
    dispatch = QualifiedCapabilityExecutionDispatchService(
        root_execution_launcher=counting_launcher,
        runtime_delegate=delegate,
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    governed_execution = _GovernedTaskExecutionAdapter(adapter, task)
    binding = QualifiedCapabilityBindingService(
        (CodeCraftQualifiedCapabilityBindingProvider(tool_wiring_context),),
    )
    authority_admission = build_worker_execution_admission_for_uca6c(
        worker_instance_id=_WORKER_ID,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    resume = WorkerQualifiedCapabilityResumeCoordinator(
        binding=binding,
        execution=governed_execution,
        authority_admission=authority_admission,
    )
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
        discovery=_missing_capability_discovery(),
        acquisition=acquisition,
        qualification=qualification,
    )
    direct_reuse = WorkerCapabilityDirectReuseFulfillmentService(
        binding=HostAvailableCapabilityBindingService(()),
        execution=governed_execution,
        authority_admission=authority_admission,
    )
    inner = WorkerCapabilityFulfillmentCoordinator(
        recovery=recovery,
        resume=resume,
        direct_reuse=direct_reuse,
        realization=None,
    )
    return (
        _RecordingFulfillmentCoordinator(inner=inner),
        counting_launcher,
        strategy,
        qualification,
        resume,
        binding,
    )


def _run_async_in_thread(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@pytest.mark.skip(
    reason="UCA-6C-R6-R5.8-R2: worker orchestration E2E wiring in progress (async/run_async + governed task seam)",
)
def test_worker_governed_execution_pause_resume_single_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in (
        "intergrax.runtime.execution.qualified_capability_execution_dispatch_service",
        "intergrax.runtime.execution.execution_bound_capability_execution_dispatch_service",
        "intergrax.tools._shared.async_dispatch",
    ):
        monkeypatch.setattr(f"{module}.run_async", _run_async_in_thread)
    task = Task(tenant_id=_TENANT, user_id="u1", message="r58-r2", task_id=_TASK_ID)
    handler, composition, _, craft_id, hitl, checkpoint_store, backend, _ = _build_handler(
        tmp_path,
        _AllowingMsePort(),
    )
    from intergrax.runtime.codecraft.wiring_bound_capability_execution import (
        WiringCodeCraftBoundCapabilityExecution,
    )

    tool_ctx = handler._execution_port._ctx  # noqa: SLF001 — test observes production wiring ctx
    assert isinstance(handler._execution_port, WiringCodeCraftBoundCapabilityExecution)
    fulfillment, root_counter, acq, qual, _, _ = _build_worker_governed_stack(
        tmp_path,
        task=task,
        tool_wiring_context=tool_ctx,
        craft_id=craft_id,
        handler=handler,
    )
    service, ctx = _harness()
    service._recovery_capability_fulfillment = WorkerRecoveryCapabilityFulfillmentService(
        fulfillment=fulfillment,
    )
    fulfillment_request = replace(
        _fulfillment_request(),
        worker_instance_id=_WORKER_ID,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    service._recovery_capability_fulfillment_request_builder = _StaticFulfillmentRequestBuilder(
        fulfillment_request,
    )
    service._execution_outcome_reader = StubExecutionOutcomeReader(
        CanonicalExecutionTerminalDisposition.IN_PROGRESS,
    )
    orch_request = _orchestration_request(
        decision=_decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY),
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        ),
    )
    task_token = bind_governed_execution_task(task)
    try:
        first = _run_async_in_thread(service.orchestrate(orch_request))
        assert first.disposition is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED, (
            first.episode.terminal_reason
        )
        execution_id = first.episode.last_execution_id
        assert execution_id is not None
        assert root_counter.launch_count == 1
        assert fulfillment.calls == 1
        assert acq.calls == 1
        assert qual.calls == 1
        assert backend.calls == 0
        worker = ctx["worker_repo"].get(worker_instance_id=first.episode.worker_instance_id)
        assert worker is not None
        assert worker.lifecycle_state is WorkerLifecycleState.WAITING_EXTERNAL

        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None and human_request is not None
        reentry = composition.suspended_work_reentry_coordinator
        assert reentry is not None
        descriptor = reentry.store.load_active_for_continuation(
            human_request.continuation_id,
        )
        assert descriptor is not None
        run_id = descriptor.identity.run_id
        attempt_id = descriptor.identity.attempt_id
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        try:
            _approve_current_pause(
                task,
                hitl=hitl,
                continuation_id=human_request.continuation_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                checkpoint_store=checkpoint_store,
            )
        finally:
            reset_active_execution_identity(id_token)

        assert backend.calls == 1
        mid = _run_async_in_thread(service.orchestrate(orch_request))
        assert mid.disposition is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED
        assert fulfillment.calls == 1
        assert root_counter.launch_count == 1
        assert backend.calls == 1

        service._execution_outcome_reader = StubExecutionOutcomeReader(
            CanonicalExecutionTerminalDisposition.SUCCEEDED,
        )
        final = _run_async_in_thread(service.orchestrate(orch_request))
        assert final.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
        assert fulfillment.calls == 1
        assert root_counter.launch_count == 1
        assert backend.calls == 1
    finally:
        reset_governed_execution_task(task_token)
        reset_active_execution_governance_identity(gov_token)
