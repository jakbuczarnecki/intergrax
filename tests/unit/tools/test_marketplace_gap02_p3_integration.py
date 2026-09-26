# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P3 bounded integration: intent + binding + canonical EE dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionRequest,
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingRequest,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.tools.qualified_capability_execution_intent_preparation import (
    QualifiedCapabilityExecutionIntentPreparationOutcome,
    QualifiedCapabilityExecutionIntentPreparationRequest,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationMaterialResult,
)
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)
from intergrax.tools.marketplace_qualified_tool_execution_intent_preparation import (
    MarketplaceQualifiedToolExecutionIntentPreparation,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_association import (
    DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_resolver import (
    MarketplaceQualifiedToolStageContextResolverImpl,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResult,
)
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)
from intergrax.tools.qualified_tool_invocation_resolver import DefaultQualifiedToolInvocationResolver
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
    _WORKER_ID,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _ACQ_REQUEST,
    _NOW,
    _QUAL_REQUEST,
    _TENANT,
    _execution_governance,
)
from tests.unit.tools.test_marketplace_qualified_capability_binding_provider import (
    _binding_stack,
    _qualification,
    _subject,
)

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_00000000000000000000000000000002")


class _Material(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = "hello"


@dataclass
class _MaterialProvider:
    def provide(self, request: QualifiedToolInvocationMaterialRequest):
        return QualifiedToolInvocationMaterialResult(
            outcome=QualifiedToolInvocationMaterialOutcome.AVAILABLE,
            material=_Material(),
        )


@dataclass
class _RecordingInvoker:
    caller_agent_id: str = "agent-gap02"
    calls: int = 0

    def invoke(self, request):
        self.calls += 1
        return ToolExecutionResult(success=True, output=_Material(), error=None)


@dataclass
class _ActivationStub:
    def ensure_exact_active(self, *, stage, execution_request_id: str):
        return QualifiedMarketplaceToolActivationResult(
            outcome=QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT,
            registry_tool_id="tool-active",
        )


def test_p3_canonical_ee_path_dispatched() -> None:
    tenant_id = _TENANT
    provider, acquisition_id, domain_ref, store = _binding_stack(
        tenant_id=tenant_id,
        acquisition_id="acq-p3",
    )
    qualification = _qualification(
        acquisition_id=acquisition_id,
        domain_ref=domain_ref,
    )
    subject = _subject(domain_ref)
    need = WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:p3",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("invoke",),
        capability_profile_ref=_PROFILE,
        requested_at=_NOW,
        recovery_decision_id="recovery:p3",
    )
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=need.recovery_decision_id,
        qualification_request_id=qualification.qualification_request_id,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    execution_request_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )

    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    intent_repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    preparation = MarketplaceQualifiedToolExecutionIntentPreparation(
        intent_repository=intent_repo,
        stage_repository=stage_repo,
        context_resolver=resolver,
    )
    prep_result = preparation.prepare(
        QualifiedCapabilityExecutionIntentPreparationRequest(
            need=need,
            qualification_result=qualification,
            execution_request_id=execution_request_id,
            binding_operation_id=binding_id,
            resume_operation_id=resume_id,
            worker_need_id="worker-need-p3",
            tenant_id=tenant_id,
            task_id=_TASK_ID,
        ),
    )
    assert prep_result.outcome is QualifiedCapabilityExecutionIntentPreparationOutcome.CREATED

    binding = QualifiedCapabilityBindingService((provider,)).bind(
        QualifiedCapabilityBindingRequest(
            binding_operation_id=binding_id,
            resume_operation_id=resume_id,
            qualified_subject=subject,
            qualification_result=qualification,
            worker_need_id="worker-need-p3",
            worker_instance_id=str(_WORKER_ID),
            tenant_id=tenant_id,
            task_id=_TASK_ID,
            requested_at=_NOW,
        ),
    )
    assert binding.execution_target is not None

    invoker = _RecordingInvoker()
    handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=intent_repo,
        stage_repository=stage_repo,
        activation_resolver=_ActivationStub(),
        material_provider=_MaterialProvider(),
        invocation_resolver=DefaultQualifiedToolInvocationResolver(),
        catalog_tool_invoker=invoker,
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    admitted, decision, scopes = _execution_governance()
    result = adapter.execute(
        WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
            execution_request_id=execution_request_id,
            execution_target=binding.execution_target,
            worker_instance_id=_WORKER_ID,
            worker_need_id="worker-need-p3",
            tenant_id=tenant_id,
            task_id=_TASK_ID,
            qualification_request_id=_QUAL_REQUEST,
            acquisition_request_id=_ACQ_REQUEST,
            qualified_subject_reference=subject.qualified_subject_reference,
            requested_at=_NOW,
            admitted_governance_identity=admitted,
            effective_authority_decision=decision,
            collaborative_authority_scopes=scopes,
        ),
    )
    assert result.disposition is WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED
    assert invoker.calls == 1
