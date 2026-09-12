# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable adapter providers — not imported by the integration engine."""

from __future__ import annotations

from dataclasses import dataclass

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


__all__ = ["SingleLifecycleAdapterProvider"]
