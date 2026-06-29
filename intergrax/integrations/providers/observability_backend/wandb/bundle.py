# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_wandb_observability_backend as _legacy_create_wandb_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.wandb.integration import (
    WANDB_OBSERVABILITY_PROVIDER_ID,
    WANDB_SUPPORTED_SIGNALS,
    WandbObservabilityIntegration,
    WandbObservabilityIntegrationConfig,
    WandbObservabilityTransport,
)

__all__ = [
    "create_wandb_observability_backend",
    "create_wandb_observability_integration",
]


def create_wandb_observability_integration(
    *,
    transport: WandbObservabilityTransport | None = None,
    enabled: bool = False,
) -> WandbObservabilityIntegration:
    """
    Build a contract-based W&B observability vendor integration.

    The legacy query facade (create_wandb_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "W&B observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return WandbObservabilityIntegration.from_transport(transport, enabled=enabled)
    return WandbObservabilityIntegration.for_provider(
        provider_id=WANDB_OBSERVABILITY_PROVIDER_ID,
        supported_signals=WANDB_SUPPORTED_SIGNALS,
        display_name="W&B",
        config=WandbObservabilityIntegrationConfig(enabled=enabled),
    )


def create_wandb_observability_backend(**kwargs: object) -> WandbObservabilityIntegration:
    """Compatibility shim — constructs WandbObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_wandb_observability_backend(**kwargs)
    if isinstance(runtime, WandbObservabilityIntegration):
        return runtime
    return WandbObservabilityIntegration.from_backend(runtime)
