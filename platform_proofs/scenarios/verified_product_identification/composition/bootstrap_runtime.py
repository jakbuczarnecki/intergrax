"""Composition root — only boundary that binds PostgreSQL and Qdrant adapters."""

from __future__ import annotations

from pathlib import Path

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    open_qdrant_vector_index_administration,
    open_qdrant_vector_store,
)

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
    artifact_directory_fingerprint,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.reader import (
    ParquetFilesystemArtifactReader,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    VpiBootstrapConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
    resolve_dataset_identity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.orchestrator import (
    VpiBootstrapDependencies,
    VpiBootstrapOrchestrator,
)
from platform_proofs.scenarios.verified_product_identification.integrations.catalog_store.postgresql.adapter import (
    PostgreSQLCatalogBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_bootstrap_adapter import (
    PlatformSearchIndexBootstrapAdapter,
)

_DENSE_CHANNEL_NAME = "dense"
_SPARSE_CHANNEL_NAME = "sparse"


def resolve_bootstrap_artifact_directory(
    config: VpiBootstrapConfig,
    *,
    artifact_dir_override: Path | None = None,
) -> Path:
    if artifact_dir_override is not None:
        return artifact_dir_override
    dataset_identity = resolve_dataset_identity(
        dataset_path=config.dataset_path,
        dataset_manifest_path=config.dataset_manifest_path,
        verification_mode=config.dataset_verification_mode,
    )
    embedding = config.embedding_configuration
    model = embedding.model
    if model is None:
        msg = "embedding model is required"
        raise ValueError(msg)
    identity = EmbeddingArtifactCompatibilityIdentity(
        dataset_checksum=dataset_identity.dataset_checksum,
        dataset_record_count=dataset_identity.dataset_record_count,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider=embedding.provider,
        embedding_model=model,
        embedding_dimension=embedding.expected_dimension,
        artifact_schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        catalog_id=config.catalog_id,
        source_revision=config.source_revision,
    )
    fingerprint = artifact_directory_fingerprint(identity)
    return config.artifact_root_dir / fingerprint


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
    artifact_dir = resolve_bootstrap_artifact_directory(config)
    embedding_artifact = ParquetFilesystemArtifactReader(artifact_dir)
    dependencies = VpiBootstrapDependencies(
        catalog=catalog,
        search=search,
        embedding_artifact=embedding_artifact,
    )
    return VpiBootstrapOrchestrator(config=config, dependencies=dependencies)
