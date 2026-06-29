# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_triton_vision_serving as _legacy_create_triton_vision_serving

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vision_serving.triton.integration import (
    TRITON_VISION_SERVING_PROVIDER_ID,
    TritonVisionServingIntegration,
    TritonVisionServingIntegrationConfig,
    TritonVisionServingClient,
)

__all__ = [
    "create_triton_vision_serving",
    "create_triton_vision_serving_integration",
]


def create_triton_vision_serving_integration(
    *,
    client: TritonVisionServingClient | None = None,
    enabled: bool = False,
) -> TritonVisionServingIntegration:
    """
    Build a contract-based Triton vision serving integration.

    The legacy facade (create_triton_vision_serving) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Triton vision serving integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TritonVisionServingIntegration.from_client(client, enabled=enabled)
    return TritonVisionServingIntegration.for_provider(
        provider_id=TRITON_VISION_SERVING_PROVIDER_ID,
        display_name="Triton",
        config=TritonVisionServingIntegrationConfig(enabled=enabled),
    )


def create_triton_vision_serving(**kwargs: object) -> TritonVisionServingIntegration:
    """Compatibility shim — constructs TritonVisionServingIntegration from legacy runtime."""
    runtime = _legacy_create_triton_vision_serving(**kwargs)
    if isinstance(runtime, TritonVisionServingIntegration):
        return runtime
    return TritonVisionServingIntegration.from_runtime(runtime)
