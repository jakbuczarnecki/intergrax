# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MLFLOW_OBSERVABILITY_PROVIDER_ID",
    "MlflowObservabilityIntegration",
    "MlflowObservabilityIntegrationConfig",
    "MlflowObservabilityTransport",
    "create_mlflow_observability_backend",
    "create_mlflow_observability_integration",
    "register_mlflow_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_mlflow_observability_backend",
        "create_mlflow_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MLFLOW_OBSERVABILITY_PROVIDER_ID",
        "MlflowObservabilityIntegration",
        "MlflowObservabilityIntegrationConfig",
        "MlflowObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_mlflow_integration":
        from intergrax.integrations.providers.observability_backend.mlflow.register import (
            register_mlflow_integration,
        )

        return register_mlflow_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.mlflow import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.mlflow import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
