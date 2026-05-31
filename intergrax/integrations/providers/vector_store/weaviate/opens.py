# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level Weaviate openers — vendor SDK imports stay in this module and ``rag_store.py``."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations._shared.p2.configs import HttpIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError


def _import_weaviate() -> Any:
    try:
        import weaviate
    except ImportError as exc:
        raise IntegrationConfigurationError("Weaviate requires weaviate-client") from exc
    return weaviate


def open_weaviate_client(config: HttpIntegrationConfig) -> Any:
    weaviate = _import_weaviate()
    url = config.require_url()
    if config.api_key:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(config.api_key),
        )
    host = url.replace("http://", "").replace("https://", "")
    return weaviate.connect_to_local(host=host)


def open_weaviate_rag_store(
    config: HttpIntegrationConfig,
    *,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    from intergrax.integrations.providers.vector_store.weaviate.rag_store import WeaviateConfig, WeaviateVectorStore

    resolved_client = client if client is not None else (client_factory() if client_factory else open_weaviate_client(config))
    rag_cfg = WeaviateConfig(collection_name=config.collection, tenant_id=config.tenant_id)
    return WeaviateVectorStore(rag_cfg, client=resolved_client)
