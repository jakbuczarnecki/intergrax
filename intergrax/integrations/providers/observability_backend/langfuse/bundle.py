# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p3.factories import create_langfuse_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.langfuse.integration import (
    LANGFUSE_OBSERVABILITY_PROVIDER_ID,
    LANGFUSE_SUPPORTED_SIGNALS,
    LangfuseObservabilityIntegration,
    LangfuseObservabilityIntegrationConfig,
    LangfuseObservabilityTransport,
)

__all__ = [
    "create_langfuse_observability_backend",
    "create_langfuse_observability_integration",
]


def create_langfuse_observability_integration(
    *,
    transport: LangfuseObservabilityTransport | None = None,
    enabled: bool = False,
) -> LangfuseObservabilityIntegration:
    """
    Build a contract-based Langfuse observability vendor integration.

    The legacy query facade (create_langfuse_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Langfuse observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return LangfuseObservabilityIntegration.from_transport(transport, enabled=enabled)
    return LangfuseObservabilityIntegration.for_provider(
        provider_id=LANGFUSE_OBSERVABILITY_PROVIDER_ID,
        supported_signals=LANGFUSE_SUPPORTED_SIGNALS,
        display_name="Langfuse",
        config=LangfuseObservabilityIntegrationConfig(enabled=enabled),
    )
