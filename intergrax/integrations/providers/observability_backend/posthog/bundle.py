# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_posthog_observability_backend as _legacy_create_posthog_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.posthog.integration import (
    POSTHOG_OBSERVABILITY_PROVIDER_ID,
    POSTHOG_SUPPORTED_SIGNALS,
    PosthogObservabilityIntegration,
    PosthogObservabilityIntegrationConfig,
    PosthogObservabilityTransport,
)

__all__ = [
    "create_posthog_observability_backend",
    "create_posthog_observability_integration",
]


def create_posthog_observability_integration(
    *,
    transport: PosthogObservabilityTransport | None = None,
    enabled: bool = False,
) -> PosthogObservabilityIntegration:
    """
    Build a contract-based Posthog observability vendor integration.

    The legacy query facade (create_posthog_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Posthog observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return PosthogObservabilityIntegration.from_transport(transport, enabled=enabled)
    return PosthogObservabilityIntegration.for_provider(
        provider_id=POSTHOG_OBSERVABILITY_PROVIDER_ID,
        supported_signals=POSTHOG_SUPPORTED_SIGNALS,
        display_name="Posthog",
        config=PosthogObservabilityIntegrationConfig(enabled=enabled),
    )


def create_posthog_observability_backend(**kwargs: object) -> PosthogObservabilityIntegration:
    """Compatibility shim — constructs PosthogObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_posthog_observability_backend(**kwargs)
    if isinstance(runtime, PosthogObservabilityIntegration):
        return runtime
    return PosthogObservabilityIntegration.from_backend(runtime)
