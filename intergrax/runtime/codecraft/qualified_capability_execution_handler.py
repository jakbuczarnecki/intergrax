# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft execution handler for bound qualified capabilities (UCA-6C-R)."""

from __future__ import annotations

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchRequest,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.contracts.execution_identity import mint_execution_id
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
    """Resolve CodeCraft execution targets — does not bypass Execution Engine dispatch."""

    def __init__(self, *, side_effect_recorder: list[str] | None = None) -> None:
        self._side_effects = side_effect_recorder

    @property
    def binding_provider_id(self) -> str:
        return CODECRAFT_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID

    def dispatch_once(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult:
        craft_id = parse_codecraft_execution_target_reference(
            request.execution_target.execution_target_reference,
        )
        if craft_id is None:
            return QualifiedCapabilityExecutionDispatchResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                execution_request_id=request.execution_request_id,
                reason_detail="invalid_codecraft_execution_target",
            )
        if self._side_effects is not None:
            self._side_effects.append(craft_id)
        return QualifiedCapabilityExecutionDispatchResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            execution_request_id=request.execution_request_id,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            execution_id=mint_execution_id(),
        )


__all__ = ["CodeCraftQualifiedCapabilityExecutionHandler"]
