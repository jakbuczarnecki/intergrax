"""Composition root — only boundary that binds PostgreSQL and Qdrant adapters."""

from __future__ import annotations

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
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant.adapter import (
    QdrantSearchIndexBootstrapAdapter,
)


def build_vpi_bootstrap_runtime(config: VpiBootstrapConfig) -> VpiBootstrapOrchestrator:
    catalog = PostgreSQLCatalogBootstrapAdapter.from_env(
        schema_name=config.postgresql_schema,
        ingestion_batch_label=config.bootstrap_implementation_version,
    )
    search = QdrantSearchIndexBootstrapAdapter.from_env(
        collection_name=config.qdrant_collection_name,
    )
    embedding = IntergraxEmbeddingBootstrapAdapter(config.embedding_configuration)
    dependencies = VpiBootstrapDependencies(
        catalog=catalog,
        search=search,
        embedding=embedding,
    )
    return VpiBootstrapOrchestrator(config=config, dependencies=dependencies)
