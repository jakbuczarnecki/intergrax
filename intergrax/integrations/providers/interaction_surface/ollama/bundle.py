# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_ollama_interaction_surface

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.interaction_surface.ollama.integration import (
    OLLAMA_INTERACTION_SURFACE_PROVIDER_ID,
    OllamaInteractionSurfaceIntegration,
    OllamaInteractionSurfaceIntegrationConfig,
    OllamaInteractionSurfaceClient,
)

__all__ = [
    "create_ollama_interaction_surface",
    "create_ollama_interaction_surface_integration",
]


def create_ollama_interaction_surface_integration(
    *,
    client: OllamaInteractionSurfaceClient | None = None,
    enabled: bool = False,
) -> OllamaInteractionSurfaceIntegration:
    """
    Build a contract-based Ollama interaction surface integration.

    The legacy facade (create_ollama_interaction_surface) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Ollama interaction surface integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OllamaInteractionSurfaceIntegration.from_client(client, enabled=enabled)
    return OllamaInteractionSurfaceIntegration.for_provider(
        provider_id=OLLAMA_INTERACTION_SURFACE_PROVIDER_ID,
        display_name="Ollama",
        config=OllamaInteractionSurfaceIntegrationConfig(enabled=enabled),
    )
