# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p5.factories import create_mlflow_observability_backend as _legacy_create_mlflow_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.mlflow.integration import (
    MLFLOW_OBSERVABILITY_PROVIDER_ID,
    MLFLOW_SUPPORTED_SIGNALS,
    MlflowObservabilityIntegration,
    MlflowObservabilityIntegrationConfig,
    MlflowObservabilityTransport,
)

__all__ = [
    "create_mlflow_observability_backend",
    "create_mlflow_observability_integration",
]


def create_mlflow_observability_integration(
    *,
    transport: MlflowObservabilityTransport | None = None,
    enabled: bool = False,
) -> MlflowObservabilityIntegration:
    """
    Build a contract-based MLflow observability vendor integration.

    The legacy query facade (create_mlflow_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "MLflow observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return MlflowObservabilityIntegration.from_transport(transport, enabled=enabled)
    return MlflowObservabilityIntegration.for_provider(
        provider_id=MLFLOW_OBSERVABILITY_PROVIDER_ID,
        supported_signals=MLFLOW_SUPPORTED_SIGNALS,
        display_name="MLflow",
        config=MlflowObservabilityIntegrationConfig(enabled=enabled),
    )


def create_mlflow_observability_backend(**kwargs: object) -> MlflowObservabilityIntegration:
    """Compatibility shim — constructs MlflowObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_mlflow_observability_backend(**kwargs)
    if isinstance(runtime, MlflowObservabilityIntegration):
        return runtime
    return MlflowObservabilityIntegration.from_backend(runtime)
