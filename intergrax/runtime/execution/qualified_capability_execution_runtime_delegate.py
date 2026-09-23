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
from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    require_active_execution_identity,
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
    ) -> None:
        self._handlers = handler_registry
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
        return handler.dispatch_once(
            dispatch_request,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )


__all__ = ["QualifiedCapabilityExecutionRuntimeDelegate"]
