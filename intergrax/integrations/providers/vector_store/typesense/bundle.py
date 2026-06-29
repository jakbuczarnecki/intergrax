# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Complete Typesense integration bundle — catalog bridge to ``intergrax/rag/``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.vector_store.typesense.integration import (
    TYPESENSE_VECTOR_STORE_PROVIDER_ID,
    TypesenseVectorStoreIntegration,
    TypesenseVectorStoreIntegrationConfig,
    TypesenseVectorStoreClient,
)
from intergrax.integrations.providers.vector_store.typesense.opens import open_typesense_vector_store


@dataclass(frozen=True)
class TypesenseIntegrationBundle:
    config: VectorIntegrationConfig
    vector_store: TypesenseVectorStoreIntegration


def resolve_typesense_config(**overrides: object) -> VectorIntegrationConfig:
    return VectorIntegrationConfig.from_env("INTERGRAX_TYPESENSE", **overrides)


def create_typesense_vector_store_integration(
    *,
    client: TypesenseVectorStoreClient | None = None,
    enabled: bool = False,
) -> TypesenseVectorStoreIntegration:
    """
    Build a contract-based Typesense vector store integration.

    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Typesense vector store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TypesenseVectorStoreIntegration.from_client(client, enabled=enabled)
    return TypesenseVectorStoreIntegration.for_provider(
        provider_id=TYPESENSE_VECTOR_STORE_PROVIDER_ID,
        display_name="Typesense",
        config=TypesenseVectorStoreIntegrationConfig(enabled=enabled),
    )


def create_typesense_integration(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> TypesenseIntegrationBundle:
    config = resolve_typesense_config(**config_overrides)
    store_impl = open_typesense_vector_store(
        config,
        config_overrides=dict(config_overrides),
        implementation=vector_store,
        store=store,
        store_factory=store_factory,
    )
    assert isinstance(store_impl, TypesenseVectorStoreIntegration)
    return TypesenseIntegrationBundle(config=config, vector_store=store_impl)


def create_typesense_vector_store(
    *,
    vector_store: Optional[VectorStore] = None,
    store: Optional[object] = None,
    store_factory: Optional[Callable[[], object]] = None,
    **config_overrides: object,
) -> TypesenseVectorStoreIntegration:
    """Compatibility shim — constructs ``TypesenseVectorStoreIntegration`` via legacy catalog path."""
    return create_typesense_integration(
        vector_store=vector_store,
        store=store,
        store_factory=store_factory,
        **config_overrides,
    ).vector_store
