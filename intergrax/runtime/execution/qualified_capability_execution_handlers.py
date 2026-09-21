# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin dispatch handlers for bound qualified capability execution targets (UCA-6C-R)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchRequest,
    QualifiedCapabilityExecutionDispatchResult,
)


@runtime_checkable
class QualifiedCapabilityExecutionBindingHandler(Protocol):
    """Domain-owned handler for one ``binding_provider_id`` — invoked by Execution Engine."""

    @property
    def binding_provider_id(self) -> str: ...

    def dispatch_once(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult: ...


class QualifiedCapabilityExecutionBindingHandlerRegistry:
    """Resolve execution handlers without AW or adapter provider branching."""

    def __init__(
        self,
        handlers: tuple[QualifiedCapabilityExecutionBindingHandler, ...],
    ) -> None:
        seen: set[str] = set()
        mapped: dict[str, QualifiedCapabilityExecutionBindingHandler] = {}
        for handler in handlers:
            provider_id = handler.binding_provider_id
            if provider_id in seen:
                raise ValueError(
                    f"duplicate qualified execution handler: {provider_id!r}"
                )
            seen.add(provider_id)
            mapped[provider_id] = handler
        self._handlers = mapped

    def resolve(
        self,
        binding_provider_id: str,
    ) -> QualifiedCapabilityExecutionBindingHandler | None:
        return self._handlers.get(binding_provider_id)


__all__ = [
    "QualifiedCapabilityExecutionBindingHandler",
    "QualifiedCapabilityExecutionBindingHandlerRegistry",
]
