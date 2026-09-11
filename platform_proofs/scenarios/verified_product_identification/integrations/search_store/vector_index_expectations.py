"""Build expected vector index identity from authoritative VPI configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.rag.vectorstore.config.vector_config import Metric

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_runtime_identity import (
    ExpectedVectorIndexRuntimeIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    CANONICAL_DISTANCE_METRIC,
    ExpectedVectorIdentity,
    expected_vector_identity_from_embedding_configuration,
)

_VPI_DATA_PACK_MANIFEST_ENV = "VPI_DATA_PACK_MANIFEST_PATH"
_VPI_EMBEDDING_MODEL_REVISION_ENV = "VPI_EMBEDDING_MODEL_REVISION"
_VPI_DATA_PACK_CONTENT_IDENTITY_ENV = "VPI_DATA_PACK_CONTENT_IDENTITY"


@dataclass(frozen=True, slots=True)
class VectorIndexExpectationSources:
    embedding_revision: str
    content_identity: str | None


def expected_vector_index_identity(
    *,
    target: VectorIndexIdentity,
    embedding_configuration: VpiEmbeddingConfiguration,
    embedding_revision: str,
    metric: Metric = CANONICAL_DISTANCE_METRIC,
    content_identity: str | None = None,
) -> ExpectedVectorIndexRuntimeIdentity:
    embedding = expected_vector_identity_from_embedding_configuration(
        embedding_configuration,
        revision=embedding_revision,
    )
    return ExpectedVectorIndexRuntimeIdentity(
        target=target,
        provider=embedding.provider,
        model=embedding.model,
        revision=embedding.revision,
        dimension=embedding.dimension,
        metric=metric,
        content_identity=content_identity,
    )


def expected_vector_index_identity_from_bootstrap_vector_identity(
    *,
    target: VectorIndexIdentity,
    embedding_identity: ExpectedVectorIdentity,
    metric: Metric = CANONICAL_DISTANCE_METRIC,
    content_identity: str | None = None,
) -> ExpectedVectorIndexRuntimeIdentity:
    return ExpectedVectorIndexRuntimeIdentity(
        target=target,
        provider=embedding_identity.provider,
        model=embedding_identity.model,
        revision=embedding_identity.revision,
        dimension=embedding_identity.dimension,
        metric=metric,
        content_identity=content_identity,
    )


def expected_vector_index_identity_from_data_pack_manifest(
    *,
    target: VectorIndexIdentity,
    manifest: DataPackManifest,
    embedding_configuration: VpiEmbeddingConfiguration,
    metric: Metric = CANONICAL_DISTANCE_METRIC,
) -> ExpectedVectorIndexRuntimeIdentity:
    return expected_vector_index_identity(
        target=target,
        embedding_configuration=embedding_configuration,
        embedding_revision=manifest.embedding_identity.resolved_model_identity(),
        metric=metric,
        content_identity=manifest.content_identity,
    )


def load_vector_index_expectation_sources() -> VectorIndexExpectationSources | None:
    manifest_path = _resolve_data_pack_manifest_path()
    if manifest_path is not None:
        manifest = read_manifest_file(manifest_path)
        return VectorIndexExpectationSources(
            embedding_revision=manifest.embedding_identity.resolved_model_identity(),
            content_identity=manifest.content_identity,
        )
    revision = os.getenv(_VPI_EMBEDDING_MODEL_REVISION_ENV, "").strip()
    content_identity = os.getenv(_VPI_DATA_PACK_CONTENT_IDENTITY_ENV, "").strip() or None
    if not revision:
        return None
    return VectorIndexExpectationSources(
        embedding_revision=revision,
        content_identity=content_identity,
    )


def build_expected_vector_index_identity_for_collection(
    *,
    collection_name: str,
    embedding_configuration: VpiEmbeddingConfiguration,
    qdrant_config: QdrantIntegrationConfig,
    expectation_sources: VectorIndexExpectationSources | None = None,
    metric: Metric = CANONICAL_DISTANCE_METRIC,
) -> ExpectedVectorIndexRuntimeIdentity:
    sources = expectation_sources or load_vector_index_expectation_sources()
    if sources is None:
        raise ValueError(
            "vector index expectation sources are unavailable; "
            "set VPI_DATA_PACK_MANIFEST_PATH or VPI_EMBEDDING_MODEL_REVISION"
        )
    target = VectorIndexIdentity(
        logical_name=collection_name,
        tenant_id=qdrant_config.tenant_id,
    )
    return expected_vector_index_identity(
        target=target,
        embedding_configuration=embedding_configuration,
        embedding_revision=sources.embedding_revision,
        metric=metric,
        content_identity=sources.content_identity,
    )


def _resolve_data_pack_manifest_path() -> Path | None:
    raw = os.getenv(_VPI_DATA_PACK_MANIFEST_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


__all__ = [
    "VectorIndexExpectationSources",
    "build_expected_vector_index_identity_for_collection",
    "expected_vector_index_identity",
    "expected_vector_index_identity_from_bootstrap_vector_identity",
    "expected_vector_index_identity_from_data_pack_manifest",
    "load_vector_index_expectation_sources",
]
