# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import (
    create_ollama_model_serving_runtime as _legacy_create_ollama_model_serving_runtime,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.model_serving_runtime.ollama.integration import (
    OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID,
    OllamaModelServingRuntimeClient,
    OllamaModelServingRuntimeIntegration,
    OllamaModelServingRuntimeIntegrationConfig,
)

__all__ = [
    "create_ollama_model_serving_runtime",
    "create_ollama_model_serving_runtime_integration",
]


def create_ollama_model_serving_runtime_integration(
    *,
    client: OllamaModelServingRuntimeClient | None = None,
    enabled: bool = False,
) -> OllamaModelServingRuntimeIntegration:
    """
    Build a contract-based Ollama model serving runtime integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Ollama model serving runtime integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OllamaModelServingRuntimeIntegration.from_client(client, enabled=enabled)
    return OllamaModelServingRuntimeIntegration.for_provider(
        provider_id=OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID,
        display_name="Ollama",
        config=OllamaModelServingRuntimeIntegrationConfig(enabled=enabled),
    )


def create_ollama_model_serving_runtime(**kwargs: object) -> OllamaModelServingRuntimeIntegration:
    """Compatibility shim — constructs OllamaModelServingRuntimeIntegration from legacy runtime."""
    runtime = _legacy_create_ollama_model_serving_runtime(**kwargs)
    if isinstance(runtime, OllamaModelServingRuntimeIntegration):
        return runtime
    return OllamaModelServingRuntimeIntegration.from_client(runtime)
