# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "WANDB_OBSERVABILITY_PROVIDER_ID",
    "WandbObservabilityIntegration",
    "WandbObservabilityIntegrationConfig",
    "WandbObservabilityTransport",
    "create_wandb_observability_backend",
    "create_wandb_observability_integration",
    "register_wandb_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_wandb_observability_backend",
        "create_wandb_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "WANDB_OBSERVABILITY_PROVIDER_ID",
        "WandbObservabilityIntegration",
        "WandbObservabilityIntegrationConfig",
        "WandbObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_wandb_integration":
        from intergrax.integrations.providers.observability_backend.wandb.register import (
            register_wandb_integration,
        )

        return register_wandb_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.wandb import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.wandb import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
