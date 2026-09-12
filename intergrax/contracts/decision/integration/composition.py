# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition contracts for Decision System integration (plugin wiring only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from intergrax.contracts.decision.integration.admission import (
    DecisionPluginAdmissionProvider,
    DefaultDecisionPluginAdmissionProvider,
)
from intergrax.contracts.decision.integration.audit import (
    DecisionIntegrationAuditProvider,
)
from intergrax.contracts.decision.integration.protocol import (
    DecisionIntegrationAdapterProvider,
)


@dataclass(frozen=True, slots=True)
class DecisionIntegrationCompositionSpec:
    """Declarative integration composition — no dynamic or untyped configuration."""

    active_lifecycle_source_types: frozenset[str]
    audit_enabled: bool

    def __post_init__(self) -> None:
        if type(self.active_lifecycle_source_types) is not frozenset:
            raise TypeError("active_lifecycle_source_types must be frozenset")
        if type(self.audit_enabled) is not bool:
            raise TypeError("audit_enabled must be bool")


@runtime_checkable
class DecisionIntegrationCompositionProvider(Protocol):
    """Supplies adapter plugins and audit for integration engine assembly."""

    @property
    def composition_spec(self) -> DecisionIntegrationCompositionSpec: ...

    @property
    def adapter_providers(self) -> tuple[DecisionIntegrationAdapterProvider, ...]: ...

    @property
    def audit_provider(self) -> DecisionIntegrationAuditProvider | None: ...

    @property
    def plugin_admission_provider(self) -> DecisionPluginAdmissionProvider: ...


@dataclass(frozen=True, slots=True)
class ConfiguredDecisionIntegrationCompositionProvider:
    """Explicit plugin wiring — used by platform composition root and tests."""

    composition_spec: DecisionIntegrationCompositionSpec
    adapter_providers: tuple[DecisionIntegrationAdapterProvider, ...]
    audit_provider: DecisionIntegrationAuditProvider | None = None
    plugin_admission_provider: DecisionPluginAdmissionProvider = field(
        default_factory=DefaultDecisionPluginAdmissionProvider,
    )

    def __post_init__(self) -> None:
        if type(self.composition_spec) is not DecisionIntegrationCompositionSpec:
            raise TypeError(
                "composition_spec must be DecisionIntegrationCompositionSpec"
            )
        if type(self.adapter_providers) is not tuple:
            raise TypeError("adapter_providers must be tuple")
        for item in self.adapter_providers:
            if not isinstance(item, DecisionIntegrationAdapterProvider):
                raise TypeError(
                    "adapter_providers items must implement DecisionIntegrationAdapterProvider",
                )
        if self.audit_provider is not None and not isinstance(
            self.audit_provider,
            DecisionIntegrationAuditProvider,
        ):
            raise TypeError(
                "audit_provider must implement DecisionIntegrationAuditProvider or be None",
            )
        if not isinstance(
            self.plugin_admission_provider,
            DecisionPluginAdmissionProvider,
        ):
            raise TypeError(
                "plugin_admission_provider must implement DecisionPluginAdmissionProvider",
            )


__all__ = [
    "ConfiguredDecisionIntegrationCompositionProvider",
    "DecisionIntegrationCompositionProvider",
    "DecisionIntegrationCompositionSpec",
]
