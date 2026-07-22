# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID",
    "OllamaModelServingRuntimeIntegration",
    "OllamaModelServingRuntimeIntegrationConfig",
    "OllamaModelServingRuntimeClient",
    "create_ollama_model_serving_runtime",
    "create_ollama_model_serving_runtime_integration",
    "register_ollama_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_ollama_model_serving_runtime",
        "create_ollama_model_serving_runtime_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID",
        "OllamaModelServingRuntimeIntegration",
        "OllamaModelServingRuntimeIntegrationConfig",
        "OllamaModelServingRuntimeClient",
    }
)


def __getattr__(name: str):
    if name == "register_ollama_integration":
        from intergrax.integrations.providers.model_serving_runtime.ollama.register import (
            register_ollama_integration,
        )

        return register_ollama_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.model_serving_runtime.ollama import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.model_serving_runtime.ollama import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
