# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_opensearch_observability_backend as _legacy_create_opensearch_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.opensearch.integration import (
    OPENSEARCH_OBSERVABILITY_PROVIDER_ID,
    OPENSEARCH_SUPPORTED_SIGNALS,
    OpensearchObservabilityIntegration,
    OpensearchObservabilityIntegrationConfig,
    OpensearchObservabilityTransport,
)

__all__ = [
    "create_opensearch_observability_backend",
    "create_opensearch_observability_integration",
]


def create_opensearch_observability_integration(
    *,
    transport: OpensearchObservabilityTransport | None = None,
    enabled: bool = False,
) -> OpensearchObservabilityIntegration:
    """
    Build a contract-based OpenSearch observability vendor integration.

    The legacy query facade (create_opensearch_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "OpenSearch observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return OpensearchObservabilityIntegration.from_transport(transport, enabled=enabled)
    return OpensearchObservabilityIntegration.for_provider(
        provider_id=OPENSEARCH_OBSERVABILITY_PROVIDER_ID,
        supported_signals=OPENSEARCH_SUPPORTED_SIGNALS,
        display_name="OpenSearch",
        config=OpensearchObservabilityIntegrationConfig(enabled=enabled),
    )


def create_opensearch_observability_backend(**kwargs: object) -> OpensearchObservabilityIntegration:
    """Compatibility shim — constructs OpensearchObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_opensearch_observability_backend(**kwargs)
    if isinstance(runtime, OpensearchObservabilityIntegration):
        return runtime
    return OpensearchObservabilityIntegration.from_backend(runtime)
