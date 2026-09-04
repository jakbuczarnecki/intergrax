"""Composition root for embedding artifact materialization."""

from __future__ import annotations

from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    VpiEmbeddingMaterializationConfig,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
    artifact_directory_fingerprint,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.orchestration.orchestrator import (
    EmbeddingMaterializationOrchestrator,
    VpiEmbeddingMaterializationDependencies,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.writer import (
    ParquetFilesystemArtifactWriter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
    resolve_dataset_identity,
)


def resolve_artifact_directory(
    config: VpiEmbeddingMaterializationConfig,
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


def build_vpi_embedding_materialization_runtime(
    config: VpiEmbeddingMaterializationConfig,
    *,
    artifact_dir: Path | None = None,
) -> EmbeddingMaterializationOrchestrator:
    resolved_artifact_dir = resolve_artifact_directory(config, artifact_dir_override=artifact_dir)
    artifact_writer = ParquetFilesystemArtifactWriter(resolved_artifact_dir)
    embedding = IntergraxEmbeddingBootstrapAdapter(config.embedding_configuration)
    dependencies = VpiEmbeddingMaterializationDependencies(
        artifact_writer=artifact_writer,
        embedding=embedding,
    )
    return EmbeddingMaterializationOrchestrator(config=config, dependencies=dependencies)
