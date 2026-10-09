# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Binding-provider execution handlers invoked inside ExecutionRuntime (UCA-6C-R2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_intake import (
    QualifiedCapabilityExecutionDelegateResult,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)


@runtime_checkable
class QualifiedCapabilityExecutionBindingHandler(Protocol):
    """Provider-owned binding execution — runs under canonical execution identity."""

    @property
    def execution_handler_id(self) -> str: ...

    def dispatch_once(
        self,
        request: BoundCapabilityExecutionDispatchRequest,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
        integration_configuration_adoption: (
            ExecutionIntegrationConfigurationAdoption | None
        ) = None,
    ) -> QualifiedCapabilityExecutionDelegateResult: ...


class QualifiedCapabilityExecutionBindingHandlerRegistry:
    """Resolve binding handlers by execution_handler_id — not an execution lifecycle owner."""

    __slots__ = ("_handlers",)

    def __init__(
        self,
        handlers: tuple[QualifiedCapabilityExecutionBindingHandler, ...],
    ) -> None:
        mapped: dict[str, QualifiedCapabilityExecutionBindingHandler] = {}
        for handler in handlers:
            handler_id = handler.execution_handler_id
            if handler_id in mapped:
                raise ValueError(
                    f"duplicate qualified capability execution handler: {handler_id}",
                )
            mapped[handler_id] = handler
        self._handlers = mapped

    def resolve(
        self,
        execution_handler_id: str,
    ) -> QualifiedCapabilityExecutionBindingHandler | None:
        return self._handlers.get(execution_handler_id)


__all__ = [
    "QualifiedCapabilityExecutionBindingHandler",
    "QualifiedCapabilityExecutionBindingHandlerRegistry",
]
