# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_signoz_observability_backend as _legacy_create_signoz_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.signoz.integration import (
    SIGNOZ_OBSERVABILITY_PROVIDER_ID,
    SIGNOZ_SUPPORTED_SIGNALS,
    SignozObservabilityIntegration,
    SignozObservabilityIntegrationConfig,
    SignozObservabilityTransport,
)

__all__ = [
    "create_signoz_observability_backend",
    "create_signoz_observability_integration",
]


def create_signoz_observability_integration(
    *,
    transport: SignozObservabilityTransport | None = None,
    enabled: bool = False,
) -> SignozObservabilityIntegration:
    """
    Build a contract-based SigNoz observability vendor integration.

    The legacy query facade (create_signoz_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "SigNoz observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return SignozObservabilityIntegration.from_transport(transport, enabled=enabled)
    return SignozObservabilityIntegration.for_provider(
        provider_id=SIGNOZ_OBSERVABILITY_PROVIDER_ID,
        supported_signals=SIGNOZ_SUPPORTED_SIGNALS,
        display_name="SigNoz",
        config=SignozObservabilityIntegrationConfig(enabled=enabled),
    )


def create_signoz_observability_backend(**kwargs: object) -> SignozObservabilityIntegration:
    """Compatibility shim — constructs SignozObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_signoz_observability_backend(**kwargs)
    if isinstance(runtime, SignozObservabilityIntegration):
        return runtime
    return SignozObservabilityIntegration.from_backend(runtime)
