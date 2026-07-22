# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ollama model serving runtime integration."""

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.model_serving_runtime import ModelServingRuntimeBackend
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.ai import ModelServingRuntimeIntegrationContract

OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID = "ollama"


class OllamaModelServingRuntimeIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Ollama model serving runtime integration."""

    pass


OllamaModelServingRuntimeClient = ModelServingRuntimeBackend


class OllamaModelServingRuntimeIntegration(ModelServingRuntimeIntegrationContract):
    """
    Single public Ollama model serving runtime entrypoint.

    Exposes host operations (``list_models``, ``health``) against a self-hosted
    Ollama server. LLM generation/embeddings remain in ``llm_adapters`` / RAG.
    """

    config: OllamaModelServingRuntimeIntegrationConfig = OllamaModelServingRuntimeIntegrationConfig()
    _client: OllamaModelServingRuntimeClient | None = PrivateAttr(default=None)

    def list_models(self) -> list[str]:
        return list(self._require_client().list_models())

    def health(self) -> HealthStatus:
        return probe_client_health(
            self._require_client(),
            slug=OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID,
            default_detail="local inference host",
        )

    def _require_client(self) -> ModelServingRuntimeBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client

    @classmethod
    def from_client(
        cls,
        client: OllamaModelServingRuntimeClient,
        *,
        enabled: bool = False,
    ) -> OllamaModelServingRuntimeIntegration:
        integration = cls.for_provider(
            provider_id=OLLAMA_MODEL_SERVING_RUNTIME_PROVIDER_ID,
            display_name="Ollama",
            config=OllamaModelServingRuntimeIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OllamaModelServingRuntimeClient | None:
        return self._client


ModelServingRuntimeBackend.register(OllamaModelServingRuntimeIntegration)
