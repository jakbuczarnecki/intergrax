# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform composition root for Decision System Integration Boundary (DS-E2E-15J).

Composition root knows default adapter plugins; the integration engine depends only on
injected abstractions via ``DecisionSystemIntegrationFactory``.
"""

from __future__ import annotations

from intergrax.contracts.decision.integration.admission import (
    DecisionPluginAdmissionProvider,
    DefaultDecisionPluginAdmissionProvider,
)
from intergrax.contracts.decision.integration.audit import (
    DefaultDecisionIntegrationAuditProvider,
    DecisionIntegrationAuditProvider,
    RecordingDecisionIntegrationAuditProvider,
)
from intergrax.contracts.decision.integration.audit_sink import (
    DecisionAuditSink,
    InMemoryDecisionAuditSink,
)
from intergrax.contracts.decision.integration.composition import (
    ConfiguredDecisionIntegrationCompositionProvider,
    DecisionIntegrationCompositionProvider,
    DecisionIntegrationCompositionSpec,
)
from intergrax.contracts.decision.integration.engine import (
    DecisionSystemIntegrationEngine,
)
from intergrax.contracts.decision.integration.factory import (
    DecisionSystemIntegrationFactory,
)
from intergrax.contracts.decision.integration.lifecycle.provider import (
    DefaultLifecycleAdapterProvider,
)
from intergrax.contracts.decision.integration.protocol import (
    DecisionIntegrationAdapterProvider,
)
from intergrax.contracts.decision.integration.references import (
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
)


def default_decision_integration_composition_provider() -> (
    DecisionIntegrationCompositionProvider
):
    """Default plugin wiring — lifecycle adapter + default audit, no global state."""
    lifecycle_provider = DefaultLifecycleAdapterProvider()
    return ConfiguredDecisionIntegrationCompositionProvider(
        composition_spec=DecisionIntegrationCompositionSpec(
            active_lifecycle_source_types=frozenset(
                {REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE},
            ),
            audit_enabled=True,
        ),
        adapter_providers=(lifecycle_provider,),
        audit_provider=DefaultDecisionIntegrationAuditProvider(),
        plugin_admission_provider=DefaultDecisionPluginAdmissionProvider(),
    )


def production_decision_integration_composition_provider(
    *,
    audit_sink: DecisionAuditSink | None = None,
    plugin_admission_provider: DecisionPluginAdmissionProvider | None = None,
) -> DecisionIntegrationCompositionProvider:
    """Production wiring — recording audit, explicit admission, lifecycle adapter."""
    sink = audit_sink if audit_sink is not None else InMemoryDecisionAuditSink()
    admission = (
        plugin_admission_provider
        if plugin_admission_provider is not None
        else DefaultDecisionPluginAdmissionProvider()
    )
    lifecycle_provider = DefaultLifecycleAdapterProvider()
    return ConfiguredDecisionIntegrationCompositionProvider(
        composition_spec=DecisionIntegrationCompositionSpec(
            active_lifecycle_source_types=frozenset(
                {REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE},
            ),
            audit_enabled=True,
        ),
        adapter_providers=(lifecycle_provider,),
        audit_provider=RecordingDecisionIntegrationAuditProvider(sink=sink),
        plugin_admission_provider=admission,
    )


def production_decision_system_integration(
    *,
    audit_sink: DecisionAuditSink | None = None,
    plugin_admission_provider: DecisionPluginAdmissionProvider | None = None,
) -> DecisionSystemIntegrationEngine:
    """Assemble production integration engine through the composition root."""
    return DecisionSystemIntegrationFactory.create_engine(
        production_decision_integration_composition_provider(
            audit_sink=audit_sink,
            plugin_admission_provider=plugin_admission_provider,
        ),
    )


def default_decision_system_integration() -> DecisionSystemIntegrationEngine:
    """Platform default integration engine assembled through the composition root."""
    return DecisionSystemIntegrationFactory.create_engine(
        default_decision_integration_composition_provider(),
    )


def compose_decision_system_integration_engine(
    composition: DecisionIntegrationCompositionProvider | None = None,
) -> DecisionSystemIntegrationEngine:
    """Compose integration engine from an explicit provider or platform defaults."""
    chosen = (
        composition
        if composition is not None
        else default_decision_integration_composition_provider()
    )
    return DecisionSystemIntegrationFactory.create_engine(chosen)


def compose_decision_system_integration_engine_with_providers(
    *,
    adapter_providers: tuple[DecisionIntegrationAdapterProvider, ...],
    audit_provider: DecisionIntegrationAuditProvider | None = None,
    active_lifecycle_source_types: frozenset[str] | None = None,
    plugin_admission_provider: DecisionPluginAdmissionProvider | None = None,
) -> DecisionSystemIntegrationEngine:
    """Explicit provider tuple — for custom platform wiring without hidden adapters."""
    source_types = (
        active_lifecycle_source_types
        if active_lifecycle_source_types is not None
        else frozenset({REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE})
    )
    provider = ConfiguredDecisionIntegrationCompositionProvider(
        composition_spec=DecisionIntegrationCompositionSpec(
            active_lifecycle_source_types=source_types,
            audit_enabled=audit_provider is not None,
        ),
        adapter_providers=adapter_providers,
        audit_provider=audit_provider,
        plugin_admission_provider=(
            plugin_admission_provider
            if plugin_admission_provider is not None
            else DefaultDecisionPluginAdmissionProvider()
        ),
    )
    return DecisionSystemIntegrationFactory.create_engine(provider)


__all__ = [
    "compose_decision_system_integration_engine",
    "compose_decision_system_integration_engine_with_providers",
    "default_decision_integration_composition_provider",
    "default_decision_system_integration",
    "production_decision_integration_composition_provider",
    "production_decision_system_integration",
]
