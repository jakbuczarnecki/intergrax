# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ExecutionRuntime delegate for bound qualified capability targets (UCA-6C-R2)."""

from __future__ import annotations

from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution.qualified_capability_execution_intake import (
    QualifiedCapabilityExecutionDelegateResult,
    QualifiedCapabilityExecutionIntakePayload,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.execution_terminal_outcome_by_execution_id import (
    record_delegate_terminal_disposition,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoptionError,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
)
from intergrax.runtime.execution.execution_integration_configuration_pinning_ports import (
    ExecutionIntegrationConfigurationExecutionPinningPort,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)


class QualifiedCapabilityExecutionRuntimeDelegate:
    """Invoke binding handlers under active ExecutionRuntime identity — not a router."""

    def __init__(
        self,
        *,
        handler_registry: QualifiedCapabilityExecutionBindingHandlerRegistry,
        terminal_outcome_store: ExecutionTerminalOutcomeByExecutionIdStore | None = None,
        integration_configuration_pinning: (
            ExecutionIntegrationConfigurationExecutionPinningPort | None
        ) = None,
    ) -> None:
        self._handlers = handler_registry
        self._terminal_outcome_store = terminal_outcome_store
        self._integration_configuration_pinning = integration_configuration_pinning
        self.execute_calls = 0

    async def execute(
        self,
        request: QualifiedCapabilityExecutionIntakePayload,
    ) -> QualifiedCapabilityExecutionDelegateResult:
        self.execute_calls += 1
        dispatch_request = BoundCapabilityExecutionDispatchRequest(
            execution_request_id=request.execution_request_id,
            execution_target=request.execution_target,
            tenant_id=request.tenant_id,
            task_id=request.task_id,
        )
        handler = self._handlers.resolve(
            request.execution_target.binding_provider_id,
        )
        if handler is None:
            return QualifiedCapabilityExecutionDelegateResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE,
                reason_detail="execution_handler_unavailable",
            )
        run_id, attempt_id = require_active_execution_identity()
        execution_id = peek_active_execution_id()
        if execution_id is None:
            return QualifiedCapabilityExecutionDelegateResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                reason_detail="active_execution_id_missing",
            )
        adoption = request.integration_configuration_adoption
        if adoption is not None:
            if self._integration_configuration_pinning is None:
                return QualifiedCapabilityExecutionDelegateResult(
                    disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                    reason_detail="integration_configuration_pinning_unavailable",
                )
            try:
                self._integration_configuration_pinning.pin_configured_adoption_for_execution(
                    tenant_id=request.tenant_id,
                    execution_id=execution_id,
                    adoption=adoption,
                )
            except (
                ExecutionIntegrationConfigurationAdoptionError,
                ExecutionIntegrationConfigurationPinningError,
            ):
                return QualifiedCapabilityExecutionDelegateResult(
                    disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                    reason_detail="integration_configuration_pinning_failed",
                )
        result = handler.dispatch_once(
            dispatch_request,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        record_delegate_terminal_disposition(
            self._terminal_outcome_store,
            execution_id=execution_id,
            disposition=result.disposition,
        )
        return result


__all__ = ["QualifiedCapabilityExecutionRuntimeDelegate"]
