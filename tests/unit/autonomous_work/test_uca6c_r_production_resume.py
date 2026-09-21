# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R — production binding, Execution Engine adapter, resume idempotency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityExecutionResult,
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeRequest,
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
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
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
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
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingRequest,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubject,
    QualifiedCapabilitySubjectKind,
    derive_qualified_subject_reference,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchRequest,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.codecraft.artifact_reference import artifact_reference_for_craft
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.ownership import CodeCraftSessionOwnership
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CodeCraftQualifiedCapabilityBindingProvider,
)
from intergrax.runtime.codecraft.qualified_capability_execution_handler import (
    CodeCraftQualifiedCapabilityExecutionHandler,
)
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_dispatch_service import (
    QualifiedCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 21, 11, 0, tzinfo=UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_TASK_ID = TaskId("task_" + "e" * 32)
_TENANT = "tenant-uca6cr"
_OTHER_TENANT = "tenant-other"
_CRAFT_ID = "craft-uca6cr-1"
_ARTIFACT = artifact_reference_for_craft(_CRAFT_ID)
_GAP = "capability-gap:need-uca6cr:missing"
_ACQ_REQUEST = "capability-acquisition-request:gap-uca6cr:nonce-1"
_QUAL_REQUEST = "capability-qualification-request:acq-uca6cr:qual-1"
_RECOVERY_DECISION = "recovery:uca6cr:1"


def _wiring(
    *,
    tenant_id: str = _TENANT,
    task_id: str = str(_TASK_ID),
    craft_id: str = _CRAFT_ID,
    register_tool: bool = True,
) -> ToolWiringContext:
    sessions = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    ownership = CodeCraftSessionOwnership(tenant_id=tenant_id, task_id=task_id)
    sessions.open(
        goal="synthesize tool",
        ownership=ownership,
        mode="autonomous",
        craft_id=craft_id,
    )
    if register_tool:
        registry.for_craft(craft_id).register("ephemeral.tool.uca6cr")
    return ToolWiringContext(
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
        },
    )


def _subject(*, artifact: str = _ARTIFACT) -> QualifiedCapabilitySubject:
    return QualifiedCapabilitySubject(
        qualification_request_id=_QUAL_REQUEST,
        subject_kind=QualifiedCapabilitySubjectKind.ARTIFACT_REFERENCE,
        subject_reference=artifact,
        qualified_subject_reference=derive_qualified_subject_reference(
            qualification_request_id=_QUAL_REQUEST,
            subject_kind=QualifiedCapabilitySubjectKind.ARTIFACT_REFERENCE,
            subject_reference=artifact,
        ),
    )


def _qualification(*, artifact: str = _ARTIFACT) -> CapabilityQualificationResult:
    return CapabilityQualificationResult(
        qualification_request_id=_QUAL_REQUEST,
        acquisition_request_id=_ACQ_REQUEST,
        gap_id=_GAP,
        strategy_id="codecraft.synthesis.v1",
        provider_id="codecraft.qualification",
        outcome=CapabilityQualificationOutcome.QUALIFIED,
        reason_code=CapabilityQualificationReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=CapabilityQualificationEvidence(
            provider_id="codecraft.qualification",
            qualification_request_id=_QUAL_REQUEST,
            acquisition_request_id=_ACQ_REQUEST,
            acquisition_strategy_id="codecraft.synthesis.v1",
            gap_id=_GAP,
            artifact_reference=artifact,
        ),
    )


def _provenance() -> WorkerCapabilityRecoveryProvenance:
    return WorkerCapabilityRecoveryProvenance(
        worker_need_id="worker-need:uca6cr:1",
        canonical_need_id="capability-need:tool:uca6cr",
        discovery_correlation_id="aw-canonical-discovery:worker-need:uca6cr:1",
        discovery_completion_outcome="missing_capability",
        gap_id=_GAP,
        acquisition_request_id=_ACQ_REQUEST,
        acquisition_strategy_id="codecraft.synthesis.v1",
        qualification_request_id=_QUAL_REQUEST,
    )


def _acquisition(*, artifact: str = _ARTIFACT) -> CapabilityAcquisitionResult:
    return CapabilityAcquisitionResult(
        request_id=_ACQ_REQUEST,
        gap_id=_GAP,
        strategy_id="codecraft.synthesis.v1",
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=CapabilityAcquisitionEvidence(artifact_reference=artifact),
    )


def _resume_request(
    qualification: CapabilityQualificationResult,
    *,
    tenant_id: str = _TENANT,
) -> WorkerQualifiedCapabilityResumeRequest:
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=_RECOVERY_DECISION,
        qualification_request_id=qualification.qualification_request_id,
    )
    return WorkerQualifiedCapabilityResumeRequest(
        worker_instance_id=_WORKER_ID,
        worker_need_id="worker-need:uca6cr:1",
        recovery_decision_id=_RECOVERY_DECISION,
        provenance=_provenance(),
        acquisition_result=_acquisition(
            artifact=qualification.evidence.artifact_reference or _ARTIFACT,
        ),
        qualification_result=qualification,
        resume_operation_id=resume_id,
        tenant_id=tenant_id,
        task_id=_TASK_ID,
        requested_at=_NOW,
    )


def _production_stack(
    ctx: ToolWiringContext,
) -> tuple[
    WorkerQualifiedCapabilityResumeCoordinator,
    QualifiedCapabilityExecutionDispatchService,
    CodeCraftQualifiedCapabilityBindingProvider,
]:
    binding_provider = CodeCraftQualifiedCapabilityBindingProvider(ctx)
    binding_service = QualifiedCapabilityBindingService((binding_provider,))
    handler_registry = QualifiedCapabilityExecutionBindingHandlerRegistry(
        (CodeCraftQualifiedCapabilityExecutionHandler(side_effect_recorder=[]),),
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=handler_registry,
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=binding_service,
        execution=execution,
    )
    return coordinator, dispatch, binding_provider


def test_codecraft_production_binding_resolves_artifact() -> None:
    ctx = _wiring()
    provider = CodeCraftQualifiedCapabilityBindingProvider(ctx)
    subject = _subject()
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=_RECOVERY_DECISION,
        qualification_request_id=_QUAL_REQUEST,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    request = QualifiedCapabilityBindingRequest(
        binding_operation_id=binding_id,
        resume_operation_id=resume_id,
        qualified_subject=subject,
        qualification_result=_qualification(),
        worker_need_id="worker-need:uca6cr:1",
        worker_instance_id=str(_WORKER_ID),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        requested_at=_NOW,
    )
    result = provider.bind(request)
    assert result.outcome is QualifiedCapabilityBindingOutcome.BOUND
    assert result.execution_target is not None


def test_stale_codecraft_artifact_binding_unavailable() -> None:
    ctx_no_session = ToolWiringContext(
        extras={
            "codecraft_session_manager": CodeCraftSessionManager(),
            "codecraft_ephemeral_registry": EphemeralToolRegistryStore(),
        },
    )
    provider = CodeCraftQualifiedCapabilityBindingProvider(ctx_no_session)
    subject = _subject()
    resume_id = "worker-capability-resume:recovery:uca6cr:stale:qual"
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    request = QualifiedCapabilityBindingRequest(
        binding_operation_id=binding_id,
        resume_operation_id=resume_id,
        qualified_subject=subject,
        qualification_result=_qualification(),
        worker_need_id="worker-need:uca6cr:1",
        worker_instance_id=str(_WORKER_ID),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        requested_at=_NOW,
    )
    result = provider.bind(request)
    assert result.outcome is QualifiedCapabilityBindingOutcome.UNAVAILABLE


def test_cross_tenant_codecraft_binding_blocked() -> None:
    ctx = _wiring()
    provider = CodeCraftQualifiedCapabilityBindingProvider(ctx)
    subject = _subject()
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=_RECOVERY_DECISION,
        qualification_request_id=_QUAL_REQUEST,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    request = QualifiedCapabilityBindingRequest(
        binding_operation_id=binding_id,
        resume_operation_id=resume_id,
        qualified_subject=subject,
        qualification_result=_qualification(),
        worker_need_id="worker-need:uca6cr:1",
        worker_instance_id=str(_WORKER_ID),
        tenant_id=_OTHER_TENANT,
        task_id=_TASK_ID,
        requested_at=_NOW,
    )
    result = provider.bind(request)
    assert result.outcome is QualifiedCapabilityBindingOutcome.BLOCKED


def test_production_ee_adapter_dispatches_once_per_request_id() -> None:
    ctx = _wiring()
    side_effects: list[str] = []
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry(
            (
                CodeCraftQualifiedCapabilityExecutionHandler(
                    side_effect_recorder=side_effects
                ),
            ),
        ),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    resume_id = "worker-capability-resume:r:e:e"
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    target = (
        CodeCraftQualifiedCapabilityBindingProvider(ctx)
        .bind(
            QualifiedCapabilityBindingRequest(
                binding_operation_id=binding_id,
                resume_operation_id=resume_id,
                qualified_subject=_subject(),
                qualification_result=_qualification(),
                worker_need_id="worker-need:uca6cr:1",
                worker_instance_id=str(_WORKER_ID),
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                requested_at=_NOW,
            ),
        )
        .execution_target
    )
    assert target is not None
    execution_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    request = WorkerQualifiedCapabilityExecutionRequest(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
        execution_request_id=execution_id,
        execution_target=target,
        worker_instance_id=_WORKER_ID,
        worker_need_id="worker-need:uca6cr:1",
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        qualification_request_id=_QUAL_REQUEST,
        acquisition_request_id=_ACQ_REQUEST,
        qualified_subject_reference=_subject().qualified_subject_reference,
        requested_at=_NOW,
    )
    adapter.execute(request)
    adapter.execute(request)
    assert dispatch.dispatch_side_effects == 1
    assert len(side_effects) == 1


def test_same_resume_exactly_once_e2e() -> None:
    ctx = _wiring()
    coordinator, dispatch, _ = _production_stack(ctx)
    request = _resume_request(_qualification())
    coordinator.resume(request)
    coordinator.resume(request)
    assert dispatch.dispatch_side_effects == 1


def test_concurrent_same_resume_single_dispatch() -> None:
    ctx = _wiring()
    coordinator, dispatch, _ = _production_stack(ctx)
    request = _resume_request(_qualification())

    def _run() -> WorkerQualifiedCapabilityResumeOutcome:
        return coordinator.resume(request).outcome

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: _run(), range(2)))
    assert all(
        item is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
        for item in outcomes
    )
    assert dispatch.dispatch_side_effects == 1


@dataclass
class _MismatchDispatch:
    def dispatch(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult:
        return QualifiedCapabilityExecutionDispatchResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            execution_request_id="wrong-execution-request-id",
        )


def test_execution_id_mismatch_fail_closed() -> None:
    ctx = _wiring()
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=WorkerQualifiedCapabilityExecutionEngineAdapter(
            dispatch=_MismatchDispatch()
        ),
    )
    result = coordinator.resume(_resume_request(_qualification()))
    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_FAILED


def test_execution_id_missing_on_dispatched_fail_closed() -> None:
    @dataclass
    class _MissingIdExecution:
        def execute(
            self,
            request: WorkerQualifiedCapabilityExecutionRequest,
        ) -> WorkerQualifiedCapabilityExecutionResult:
            return WorkerQualifiedCapabilityExecutionResult(
                disposition=WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED,
                execution_request_id=None,
            )

    ctx = _wiring()
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=_MissingIdExecution(),
    )
    with pytest.raises(ValueError, match="DISPATCHED requires execution_request_id"):
        coordinator.resume(_resume_request(_qualification()))
