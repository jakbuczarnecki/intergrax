# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_braintrust_observability_backend as _legacy_create_braintrust_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.braintrust.integration import (
    BRAINTRUST_OBSERVABILITY_PROVIDER_ID,
    BRAINTRUST_SUPPORTED_SIGNALS,
    BraintrustObservabilityIntegration,
    BraintrustObservabilityIntegrationConfig,
    BraintrustObservabilityTransport,
)

__all__ = [
    "create_braintrust_observability_backend",
    "create_braintrust_observability_integration",
]


def create_braintrust_observability_integration(
    *,
    transport: BraintrustObservabilityTransport | None = None,
    enabled: bool = False,
) -> BraintrustObservabilityIntegration:
    """
    Build a contract-based Braintrust observability vendor integration.

    The legacy query facade (create_braintrust_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Braintrust observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return BraintrustObservabilityIntegration.from_transport(transport, enabled=enabled)
    return BraintrustObservabilityIntegration.for_provider(
        provider_id=BRAINTRUST_OBSERVABILITY_PROVIDER_ID,
        supported_signals=BRAINTRUST_SUPPORTED_SIGNALS,
        display_name="Braintrust",
        config=BraintrustObservabilityIntegrationConfig(enabled=enabled),
    )


def create_braintrust_observability_backend(**kwargs: object) -> BraintrustObservabilityIntegration:
    """Compatibility shim — constructs BraintrustObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_braintrust_observability_backend(**kwargs)
    if isinstance(runtime, BraintrustObservabilityIntegration):
        return runtime
    return BraintrustObservabilityIntegration.from_backend(runtime)
