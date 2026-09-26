# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace Tool qualified capability execution handler (S24-GAP-02-P3)."""

from __future__ import annotations

from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution.qualified_capability_execution_intake import (
    QualifiedCapabilityExecutionDelegateResult,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStageIntegrityError,
    MarketplaceQualifiedToolStageRepository,
    MarketplaceQualifiedToolStageUnavailableError,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntentIntegrityError,
    QualifiedMarketplaceToolExecutionIntentRepository,
    QualifiedMarketplaceToolExecutionIntentUnavailableError,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialProvider,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationResolver,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandler,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
    parse_marketplace_qualified_tool_execution_target_reference,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResolver,
)


class MarketplaceToolQualifiedCapabilityExecutionHandler(
    QualifiedCapabilityExecutionBindingHandler,
):
    """Canonical EE handler — ToolRuntime invocation only."""

    def __init__(
        self,
        *,
        intent_repository: QualifiedMarketplaceToolExecutionIntentRepository,
        stage_repository: MarketplaceQualifiedToolStageRepository,
        activation_resolver: QualifiedMarketplaceToolActivationResolver,
        material_provider: QualifiedToolInvocationMaterialProvider,
        invocation_resolver: QualifiedToolInvocationResolver,
        catalog_tool_invoker: ExecutionBoundCatalogToolInvoker,
    ) -> None:
        self._intent_repository = intent_repository
        self._stage_repository = stage_repository
        self._activation_resolver = activation_resolver
        self._material_provider = material_provider
        self._invocation_resolver = invocation_resolver
        self._catalog_tool_invoker = catalog_tool_invoker

    @property
    def binding_provider_id(self) -> str:
        return MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID

    def dispatch_once(
        self,
        request: BoundCapabilityExecutionDispatchRequest,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> QualifiedCapabilityExecutionDelegateResult:
        _ = attempt_id
        target = request.execution_target
        if target.binding_provider_id != self.binding_provider_id:
            return _failed("binding_provider_mismatch")

        handoff_id = parse_marketplace_qualified_tool_execution_target_reference(
            target.execution_target_reference,
        )
        if handoff_id is None:
            return _failed("invalid_execution_target_reference")

        try:
            intent = self._intent_repository.get(
                execution_request_id=request.execution_request_id,
            )
        except QualifiedMarketplaceToolExecutionIntentUnavailableError:
            return _unavailable("intent_store_unavailable")
        except QualifiedMarketplaceToolExecutionIntentIntegrityError:
            return _failed("intent_corrupt")

        if intent is None:
            return _failed("intent_not_found")

        if intent.tenant_id != request.tenant_id:
            return _failed("intent_tenant_mismatch")
        if intent.task_id != str(request.task_id):
            return _failed("intent_task_mismatch")
        if intent.qualified_subject_reference != target.qualified_subject_reference:
            return _failed("intent_subject_mismatch")
        if intent.handoff_id != handoff_id:
            return _failed("intent_handoff_mismatch")

        try:
            stage = self._stage_repository.get(
                tenant_id=request.tenant_id,
                handoff_id=handoff_id,
            )
        except MarketplaceQualifiedToolStageUnavailableError:
            return _unavailable("stage_store_unavailable")
        except MarketplaceQualifiedToolStageIntegrityError:
            return _failed("stage_corrupt")

        if stage is None:
            return _failed("stage_missing")

        activation = self._activation_resolver.ensure_exact_active(
            stage=stage,
            execution_request_id=request.execution_request_id,
        )
        activation_outcome = activation.outcome
        if activation_outcome is QualifiedMarketplaceToolActivationOutcome.UNAVAILABLE:
            return _unavailable(activation.reason_detail or "activation_unavailable")
        if activation_outcome in {
            QualifiedMarketplaceToolActivationOutcome.RELEASE_CONFLICT,
            QualifiedMarketplaceToolActivationOutcome.RESOLUTION_FAILURE,
            QualifiedMarketplaceToolActivationOutcome.ACTIVATION_FAILURE,
            QualifiedMarketplaceToolActivationOutcome.INTEGRITY_FAILURE,
        }:
            return _failed(activation.reason_detail or "activation_failed")
        if activation_outcome not in {
            QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT,
            QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT,
        }:
            return _failed("activation_unexpected_outcome")

        registry_tool_id = activation.registry_tool_id
        if not registry_tool_id:
            return _failed("activation_missing_registry_tool_id")

        material_result = self._material_provider.provide(
            QualifiedToolInvocationMaterialRequest(
                execution_request_id=request.execution_request_id,
                tenant_id=request.tenant_id,
                task_id=request.task_id,
                selected_operation=intent.selected_operation,
                qualified_subject_reference=intent.qualified_subject_reference,
                handoff_id=intent.handoff_id,
                worker_need_id=intent.worker_need_id,
                activated_tool_id=registry_tool_id,
            ),
        )
        if material_result.outcome is QualifiedToolInvocationMaterialOutcome.UNAVAILABLE:
            return _unavailable(material_result.reason_detail or "material_unavailable")
        if material_result.outcome is QualifiedToolInvocationMaterialOutcome.INVALID:
            return _failed(material_result.reason_detail or "material_invalid")
        material = material_result.material
        if material is None:
            return _failed("material_missing")

        step_id = f"qmte:{request.execution_request_id}"
        invoke_request = self._invocation_resolver.resolve(
            activated_tool_id=registry_tool_id,
            selected_operation=intent.selected_operation,
            material=material,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
            run_id=str(run_id),
            agent_id=self._catalog_tool_invoker.caller_agent_id,
            step_id=step_id,
            execution_request_id=request.execution_request_id,
            correlation_request_id=str(execution_id),
            idempotency_key=f"qmte:{request.execution_request_id}:{intent.selected_operation}",
        )

        try:
            tool_result = self._catalog_tool_invoker.invoke(invoke_request)
        except ExecutionSuspendedWorkPauseRequired:
            raise

        if tool_result.success:
            return QualifiedCapabilityExecutionDelegateResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            )
        return _failed("tool_execution_failed")


def _failed(detail: str) -> QualifiedCapabilityExecutionDelegateResult:
    return QualifiedCapabilityExecutionDelegateResult(
        disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
        reason_detail=detail,
    )


def _unavailable(detail: str) -> QualifiedCapabilityExecutionDelegateResult:
    return QualifiedCapabilityExecutionDelegateResult(
        disposition=QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE,
        reason_detail=detail,
    )


__all__ = ["MarketplaceToolQualifiedCapabilityExecutionHandler"]
