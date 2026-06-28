# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_langsmith_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.langsmith.integration import (
    LANGSMITH_OBSERVABILITY_PROVIDER_ID,
    LANGSMITH_SUPPORTED_SIGNALS,
    LangsmithObservabilityIntegration,
    LangsmithObservabilityIntegrationConfig,
    LangsmithObservabilityTransport,
)

__all__ = [
    "create_langsmith_observability_backend",
    "create_langsmith_observability_integration",
]


def create_langsmith_observability_integration(
    *,
    transport: LangsmithObservabilityTransport | None = None,
    enabled: bool = False,
) -> LangsmithObservabilityIntegration:
    """
    Build a contract-based Langsmith observability vendor integration.

    The legacy query facade (create_langsmith_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Langsmith observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return LangsmithObservabilityIntegration.from_transport(transport, enabled=enabled)
    return LangsmithObservabilityIntegration.for_provider(
        provider_id=LANGSMITH_OBSERVABILITY_PROVIDER_ID,
        supported_signals=LANGSMITH_SUPPORTED_SIGNALS,
        display_name="Langsmith",
        config=LangsmithObservabilityIntegrationConfig(enabled=enabled),
    )
