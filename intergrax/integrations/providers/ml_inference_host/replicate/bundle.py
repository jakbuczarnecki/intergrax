# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_replicate_ml_inference_host as _legacy_create_replicate_ml_inference_host

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.ml_inference_host.replicate.integration import (
    REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID,
    ReplicateMlInferenceHostIntegration,
    ReplicateMlInferenceHostIntegrationConfig,
    ReplicateMlInferenceHostClient,
)

__all__ = [
    "create_replicate_ml_inference_host",
    "create_replicate_ml_inference_host_integration",
]


def create_replicate_ml_inference_host_integration(
    *,
    client: ReplicateMlInferenceHostClient | None = None,
    enabled: bool = False,
) -> ReplicateMlInferenceHostIntegration:
    """
    Build a contract-based Replicate ml inference host integration.

    The legacy facade (create_replicate_ml_inference_host) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Replicate ml inference host integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ReplicateMlInferenceHostIntegration.from_client(client, enabled=enabled)
    return ReplicateMlInferenceHostIntegration.for_provider(
        provider_id=REPLICATE_ML_INFERENCE_HOST_PROVIDER_ID,
        display_name="Replicate",
        config=ReplicateMlInferenceHostIntegrationConfig(enabled=enabled),
    )


def create_replicate_ml_inference_host(**kwargs: object) -> ReplicateMlInferenceHostIntegration:
    """Compatibility shim — constructs ReplicateMlInferenceHostIntegration from legacy runtime."""
    runtime = _legacy_create_replicate_ml_inference_host(**kwargs)
    if isinstance(runtime, ReplicateMlInferenceHostIntegration):
        return runtime
    return ReplicateMlInferenceHostIntegration.from_client(runtime)
