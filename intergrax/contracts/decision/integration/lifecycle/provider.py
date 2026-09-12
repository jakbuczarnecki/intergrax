# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable adapter providers — not imported by the integration engine."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision.integration.lifecycle.default_adapter import (
    DEFAULT_LIFECYCLE_ADAPTER_ID,
    DEFAULT_LIFECYCLE_ADAPTER_VERSION,
    DefaultDecisionLifecycleIntegrationAdapter,
)
from intergrax.contracts.decision.integration.metadata import (
    DecisionIntegrationPluginDescriptor,
)
from intergrax.contracts.decision.integration.protocol import (
    DecisionLifecycleIntegrationAdapter,
)

_LIFECYCLE_ADAPTER_PLUGIN_SOURCE = "decision.integration.lifecycle_adapter"


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

    def integration_plugin_descriptor(self) -> DecisionIntegrationPluginDescriptor:
        return DecisionIntegrationPluginDescriptor(
            plugin_id=self.lifecycle_adapter.adapter_id,
            version=self.lifecycle_adapter.adapter_version,
            source=_LIFECYCLE_ADAPTER_PLUGIN_SOURCE,
            manifest_id=None,
        )


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

    def integration_plugin_descriptor(self) -> DecisionIntegrationPluginDescriptor:
        return DecisionIntegrationPluginDescriptor(
            plugin_id=DEFAULT_LIFECYCLE_ADAPTER_ID,
            version=DEFAULT_LIFECYCLE_ADAPTER_VERSION,
            source=_LIFECYCLE_ADAPTER_PLUGIN_SOURCE,
            manifest_id=None,
        )


__all__ = ["DefaultLifecycleAdapterProvider", "SingleLifecycleAdapterProvider"]
