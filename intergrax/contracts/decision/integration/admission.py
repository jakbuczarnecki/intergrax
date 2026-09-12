# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin admission for decision integration adapter providers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.decision.integration.metadata import (
    DecisionIntegrationPluginDescriptor,
)
from intergrax.contracts.decision.integration.protocol import (
    DecisionIntegrationAdapterProvider,
)


class PluginAdmissionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW_REQUIRED = "review_required"


@runtime_checkable
class DecisionIntegrationPluginIdentifiable(Protocol):
    def integration_plugin_descriptor(self) -> DecisionIntegrationPluginDescriptor: ...


@runtime_checkable
class DecisionPluginAdmissionProvider(Protocol):
    def evaluate(
        self,
        descriptor: DecisionIntegrationPluginDescriptor,
    ) -> PluginAdmissionDecision: ...


@dataclass(frozen=True, slots=True)
class DefaultDecisionPluginAdmissionProvider:
    """Default admission plugin — allows all descriptors until policy replaces it."""

    def evaluate(
        self,
        descriptor: DecisionIntegrationPluginDescriptor,
    ) -> PluginAdmissionDecision:
        if type(descriptor) is not DecisionIntegrationPluginDescriptor:
            raise TypeError("descriptor must be DecisionIntegrationPluginDescriptor")
        return PluginAdmissionDecision.ALLOW


def resolve_integration_plugin_descriptor(
    provider: DecisionIntegrationAdapterProvider,
) -> DecisionIntegrationPluginDescriptor:
    if isinstance(provider, DecisionIntegrationPluginIdentifiable):
        return provider.integration_plugin_descriptor()
    return DecisionIntegrationPluginDescriptor(
        plugin_id=type(provider).__name__,
        version="0",
        source="decision.integration.adapter_provider",
        manifest_id=None,
    )


def filter_admitted_adapter_providers(
    providers: tuple[DecisionIntegrationAdapterProvider, ...],
    admission_provider: DecisionPluginAdmissionProvider,
) -> tuple[DecisionIntegrationAdapterProvider, ...]:
    if not isinstance(admission_provider, DecisionPluginAdmissionProvider):
        raise TypeError(
            "admission_provider must implement DecisionPluginAdmissionProvider",
        )
    admitted: list[DecisionIntegrationAdapterProvider] = []
    for provider in providers:
        descriptor = resolve_integration_plugin_descriptor(provider)
        decision = admission_provider.evaluate(descriptor)
        if decision is PluginAdmissionDecision.ALLOW:
            admitted.append(provider)
    return tuple(admitted)


__all__ = [
    "DecisionIntegrationPluginIdentifiable",
    "DecisionPluginAdmissionProvider",
    "DefaultDecisionPluginAdmissionProvider",
    "PluginAdmissionDecision",
    "filter_admitted_adapter_providers",
    "resolve_integration_plugin_descriptor",
]
