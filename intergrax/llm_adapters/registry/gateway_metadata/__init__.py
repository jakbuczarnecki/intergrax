# © Artur Czarnecki. All rights reserved.

"""Optional gateway model metadata fetch + session cache (M-LLM-X.14.2 · ADR-LLM-002)."""

from intergrax.llm_adapters.registry.gateway_metadata.openrouter_client import (
    GatewayModelMetadata,
    OpenRouterModelMetadataClient,
)
from intergrax.llm_adapters.registry.gateway_metadata.session import (
    lookup_gateway_context_window,
    reset_gateway_metadata_session,
)

__all__ = [
    "GatewayModelMetadata",
    "OpenRouterModelMetadataClient",
    "lookup_gateway_context_window",
    "reset_gateway_metadata_session",
]
