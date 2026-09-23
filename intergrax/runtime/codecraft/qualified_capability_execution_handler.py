# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft binding handler under ExecutionRuntime identity (UCA-6C-R2)."""

from __future__ import annotations

from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionPort,
    CodeCraftBoundCapabilityExecutionRequest,
    CodeCraftBoundCapabilityExecutionResult,
)
from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution.qualified_capability_execution_intake import (
    QualifiedCapabilityExecutionDelegateResult,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.runtime.codecraft.artifact_reference import (
    parse_codecraft_execution_target_reference,
)
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CODECRAFT_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandler,
)


class CodeCraftQualifiedCapabilityExecutionHandler(
    QualifiedCapabilityExecutionBindingHandler,
):
    """Resolve CodeCraft execution targets inside canonical ExecutionRuntime."""

    def __init__(
        self,
        *,
        execution_port: CodeCraftBoundCapabilityExecutionPort,
        side_effect_recorder: list[str] | None = None,
    ) -> None:
        self._execution_port = execution_port
        self._side_effects = side_effect_recorder

    @property
    def binding_provider_id(self) -> str:
        return CODECRAFT_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID

    def dispatch_once(
        self,
        request: BoundCapabilityExecutionDispatchRequest,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> QualifiedCapabilityExecutionDelegateResult:
        craft_id = parse_codecraft_execution_target_reference(
            request.execution_target.execution_target_reference,
        )
        if craft_id is None:
            return QualifiedCapabilityExecutionDelegateResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                reason_detail="invalid_codecraft_execution_target",
            )

        _ = (run_id, attempt_id)
        try:
            port_result = self._execution_port.execute(
                CodeCraftBoundCapabilityExecutionRequest(
                    craft_id=craft_id,
                    tenant_id=request.tenant_id,
                    task_id=request.task_id,
                    run_id=None,
                    execution_id=execution_id,
                    execution_request_id=request.execution_request_id,
                ),
            )
        except ExecutionSuspendedWorkPauseRequired:
            raise
        if (
            port_result.outcome is CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED
            and self._side_effects is not None
        ):
            self._side_effects.append(craft_id)
        return _map_port_result(port_result)


def _map_port_result(
    port_result: CodeCraftBoundCapabilityExecutionResult,
) -> QualifiedCapabilityExecutionDelegateResult:
    outcome = port_result.outcome
    detail = port_result.reason_detail
    if outcome is CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED:
        return QualifiedCapabilityExecutionDelegateResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            reason_detail=detail,
        )
    if outcome is CodeCraftBoundCapabilityExecutionOutcome.REJECTED:
        return QualifiedCapabilityExecutionDelegateResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.REJECTED,
            reason_detail=detail or "codecraft_execution_rejected",
        )
    if outcome is CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE:
        return QualifiedCapabilityExecutionDelegateResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE,
            reason_detail=detail or "codecraft_execution_unavailable",
        )
    return QualifiedCapabilityExecutionDelegateResult(
        disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
        reason_detail=detail or "codecraft_execution_failed",
    )


__all__ = ["CodeCraftQualifiedCapabilityExecutionHandler"]
