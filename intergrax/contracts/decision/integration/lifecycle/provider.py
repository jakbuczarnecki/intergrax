# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable adapter providers — not imported by the integration engine."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision.integration.lifecycle.default_adapter import (
    DefaultDecisionLifecycleIntegrationAdapter,
)
from intergrax.contracts.decision.integration.protocol import (
    DecisionLifecycleIntegrationAdapter,
)


@dataclass(frozen=True, slots=True)
class SingleLifecycleAdapterProvider:
    """Registers exactly one lifecycle adapter implementation."""

    lifecycle_adapter: DecisionLifecycleIntegrationAdapter

    def __post_init__(self) -> None:
        if not isinstance(self.lifecycle_adapter, DecisionLifecycleIntegrationAdapter):
            raise TypeError(
                "lifecycle_adapter must implement DecisionLifecycleIntegrationAdapter",
            )

    def provide_lifecycle_adapter(
        self,
        source_type: str,
    ) -> DecisionLifecycleIntegrationAdapter | None:
        if source_type == self.lifecycle_adapter.source_type:
            return self.lifecycle_adapter
        return None


@dataclass(frozen=True, slots=True)
class DefaultLifecycleAdapterProvider:
    """Default lifecycle adapter plugin — swappable at composition root."""

    _inner: SingleLifecycleAdapterProvider

    def __init__(self) -> None:
        inner = SingleLifecycleAdapterProvider(
            lifecycle_adapter=DefaultDecisionLifecycleIntegrationAdapter(),
        )
        object.__setattr__(self, "_inner", inner)

    def provide_lifecycle_adapter(
        self,
        source_type: str,
    ) -> DecisionLifecycleIntegrationAdapter | None:
        return self._inner.provide_lifecycle_adapter(source_type)


__all__ = ["DefaultLifecycleAdapterProvider", "SingleLifecycleAdapterProvider"]
