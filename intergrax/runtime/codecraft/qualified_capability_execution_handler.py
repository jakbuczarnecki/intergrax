# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft binding handler under ExecutionRuntime identity (UCA-6C-R2)."""

from __future__ import annotations

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchRequest,
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
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandler,
)


class CodeCraftQualifiedCapabilityExecutionHandler(
    QualifiedCapabilityExecutionBindingHandler,
):
    """Resolve CodeCraft execution targets inside canonical ExecutionRuntime."""

    def __init__(self, *, side_effect_recorder: list[str] | None = None) -> None:
        self._side_effects = side_effect_recorder

    @property
    def binding_provider_id(self) -> str:
        return CODECRAFT_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID

    def dispatch_once(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
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
        if self._side_effects is not None:
            self._side_effects.append(craft_id)
        _ = (run_id, attempt_id, execution_id)
        return QualifiedCapabilityExecutionDelegateResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
        )


__all__ = ["CodeCraftQualifiedCapabilityExecutionHandler"]
