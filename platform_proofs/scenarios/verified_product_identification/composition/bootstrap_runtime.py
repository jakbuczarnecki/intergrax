"""Composition root — only boundary that binds PostgreSQL and Qdrant adapters."""

from __future__ import annotations

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    open_qdrant_vector_index_administration,
    open_qdrant_vector_store,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    VpiBootstrapConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.orchestrator import (
    VpiBootstrapDependencies,
    VpiBootstrapOrchestrator,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.catalog_store.postgresql.adapter import (
    PostgreSQLCatalogBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_bootstrap_adapter import (
    PlatformSearchIndexBootstrapAdapter,
)

_DENSE_CHANNEL_NAME = "dense"
_SPARSE_CHANNEL_NAME = "sparse"


def _open_reference_qdrant_search_adapter(
    config: VpiBootstrapConfig,
) -> PlatformSearchIndexBootstrapAdapter:
    """Reference composition: Qdrant control + data plane via public platform openers."""
    qdrant_config = QdrantIntegrationConfig.from_env(
        collection_name=config.qdrant_collection_name,
        enable_sparse_vectors=True,
    )
    index_identity = VectorIndexIdentity(
        logical_name=config.qdrant_collection_name,
        tenant_id=qdrant_config.tenant_id,
    )
    return PlatformSearchIndexBootstrapAdapter(
        _index_admin=open_qdrant_vector_index_administration(qdrant_config),
        _vector_store=open_qdrant_vector_store(qdrant_config),
        _index_identity=index_identity,
        _dense_channel_name=_DENSE_CHANNEL_NAME,
        _sparse_channel_name=_SPARSE_CHANNEL_NAME,
        _sparse_required=qdrant_config.enable_sparse_vectors,
    )


def build_vpi_bootstrap_runtime(config: VpiBootstrapConfig) -> VpiBootstrapOrchestrator:
    catalog = PostgreSQLCatalogBootstrapAdapter.from_env(
        schema_name=config.postgresql_schema,
        ingestion_batch_label=config.bootstrap_implementation_version,
    )
    search = _open_reference_qdrant_search_adapter(config)
    embedding = IntergraxEmbeddingBootstrapAdapter(config.embedding_configuration)
    dependencies = VpiBootstrapDependencies(
        catalog=catalog,
        search=search,
        embedding=embedding,
    )
    return VpiBootstrapOrchestrator(config=config, dependencies=dependencies)
