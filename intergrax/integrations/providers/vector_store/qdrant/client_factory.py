# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal Qdrant client factory shared by public openers."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.index_administration import (
    QdrantControlPlaneClient,
)


def _import_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise IntegrationConfigurationError(
            "Qdrant integration requires qdrant-client. "
            "Install with: Intergrax-ai[vector-qdrant]."
        ) from exc
    return QdrantClient


def build_qdrant_control_plane_client(
    config: QdrantIntegrationConfig,
) -> QdrantControlPlaneClient:
    QdrantClient = _import_qdrant_client()
    if config.resolved_url():
        return QdrantClient(url=config.resolved_url(), api_key=config.api_key or None)
    return QdrantClient(host=config.host, port=config.port, api_key=config.api_key or None)
