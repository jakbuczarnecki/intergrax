# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntent,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationMaterialResult,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.tools.execution_models import ToolExecutionError, ToolExecutionResult
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
    execution_target_reference_for_marketplace_qualified_tool,
)
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResult,
    QualifiedMarketplaceToolActivationResolver,
)
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)
from intergrax.tools.qualified_tool_invocation_resolver import DefaultQualifiedToolInvocationResolver
from tests.unit.tools.test_marketplace_qualified_capability_binding_provider import _release

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_00000000000000000000000000000001")
_RUN_ID = RunId("run_" + "a" * 32)
_ATTEMPT_ID = AttemptId("attempt_" + "b" * 28)
_EXECUTION_ID = ExecutionId("execution_" + "c" * 24)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = "ok"


@dataclass
class _MaterialProvider:
    def provide(self, request: QualifiedToolInvocationMaterialRequest):
        return QualifiedToolInvocationMaterialResult(
            outcome=QualifiedToolInvocationMaterialOutcome.AVAILABLE,
            material=_Input(),
        )


@dataclass
class _Invoker:
    caller_agent_id: str = "agent-caller"
    calls: int = 0
    suspend: bool = False
    success: bool = True

    def invoke(self, request):
        self.calls += 1
        if self.suspend:
            raise ExecutionSuspendedWorkPauseRequired.__new__(
                ExecutionSuspendedWorkPauseRequired,
            )
        return ToolExecutionResult(
            success=self.success,
            output=_Input() if self.success else None,
            error=None if self.success else ToolExecutionError("failed", "failed"),
        )


@dataclass
class _ActivationResolver:
    outcome: QualifiedMarketplaceToolActivationOutcome = (
        QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT
    )
    registry_tool_id: str = "tool-active"

    def ensure_exact_active(self, *, stage, execution_request_id: str):
        return QualifiedMarketplaceToolActivationResult(
            outcome=self.outcome,
            registry_tool_id=self.registry_tool_id,
        )


def _handler(
    *,
    intent: QualifiedMarketplaceToolExecutionIntent | None,
    activation: _ActivationResolver | None = None,
    invoker: _Invoker | None = None,
) -> tuple[MarketplaceToolQualifiedCapabilityExecutionHandler, _Invoker]:
    store = InMemoryDocumentStore()
    intent_repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    if intent is not None:
        intent_repo.record(intent)
        from intergrax.contracts.marketplace.handoff_traceability import (
            CapabilityHandoffConsumerTarget,
        )
        from intergrax.contracts.tools.marketplace_qualified_capability import (
            MarketplaceQualifiedToolStage,
        )

        stage_repo.stage(
            MarketplaceQualifiedToolStage(
                handoff_id=intent.handoff_id,
                tenant_id=intent.tenant_id,
                selected_release=_release(),
                discovery_correlation_id="discovery-1",
                selection_id="selection-1",
                consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
                downstream_consumer_id="tool.qualification_staging.v1",
                recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=UTC),
            ),
        )
    inv = invoker or _Invoker()
    handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=intent_repo,
        stage_repository=stage_repo,
        activation_resolver=activation or _ActivationResolver(),
        material_provider=_MaterialProvider(),
        invocation_resolver=DefaultQualifiedToolInvocationResolver(),
        catalog_tool_invoker=inv,
    )
    return handler, inv


def _intent() -> QualifiedMarketplaceToolExecutionIntent:
    return QualifiedMarketplaceToolExecutionIntent(
        execution_request_id="exec-req-1",
        binding_operation_id="bind-1",
        resume_operation_id="resume-1",
        tenant_id="tenant-1",
        task_id=str(_TASK_ID),
        worker_need_id="worker-need-1",
        qualified_subject_reference="qualified-capability-subject:q:domain_handoff_reference:h",
        handoff_id="handoff-1",
        selected_operation="invoke",
    )


def _dispatch(target: QualifiedCapabilityExecutionTarget) -> BoundCapabilityExecutionDispatchRequest:
    return BoundCapabilityExecutionDispatchRequest(
        execution_request_id="exec-req-1",
        execution_target=target,
        tenant_id="tenant-1",
        task_id=_TASK_ID,
    )


def test_success_dispatched_invoker_called_once() -> None:
    target = QualifiedCapabilityExecutionTarget(
        execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
            "handoff-1",
        ),
        binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
        qualified_subject_reference=_intent().qualified_subject_reference,
    )
    handler, invoker = _handler(intent=_intent())
    result = handler.dispatch_once(
        _dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
    assert invoker.calls == 1


def test_intent_missing_failed() -> None:
    target = QualifiedCapabilityExecutionTarget(
        execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
            "handoff-1",
        ),
        binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
        qualified_subject_reference=_intent().qualified_subject_reference,
    )
    handler, invoker = _handler(intent=None)
    result = handler.dispatch_once(
        _dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert invoker.calls == 0


def test_suspension_propagates() -> None:
    target = QualifiedCapabilityExecutionTarget(
        execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
            "handoff-1",
        ),
        binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
        qualified_subject_reference=_intent().qualified_subject_reference,
    )
    handler, _ = _handler(intent=_intent(), invoker=_Invoker(suspend=True))
    with pytest.raises(ExecutionSuspendedWorkPauseRequired):
        handler.dispatch_once(
            _dispatch(target),
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
        )
