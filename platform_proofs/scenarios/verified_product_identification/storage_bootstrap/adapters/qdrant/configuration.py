"""Typed Qdrant configuration for VPI vector storage bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.vector_store.qdrant.config import (
    QdrantIntegrationConfig,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)

CANONICAL_EMBEDDING_PROVIDER = "hf"
CANONICAL_EMBEDDING_MODEL = "BAAI/bge-m3"
CANONICAL_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CANONICAL_EMBEDDING_DIMENSION = 1024
CANONICAL_DISTANCE_METRIC = "cosine"

DEFAULT_LOGICAL_COLLECTION_NAME = "vpi-product-embeddings"
DEFAULT_UPSERT_BATCH_SIZE = 64

# Float32 transport equality — Qdrant stores dense vectors as float32.
VECTOR_TRANSPORT_FLOAT32_TOLERANCE = 0.0


@dataclass(frozen=True, slots=True)
class ExpectedVectorIdentity:
    provider: str
    model: str
    revision: str
    dimension: int

    @classmethod
    def canonical_vpi(cls) -> ExpectedVectorIdentity:
        return cls(
            provider=CANONICAL_EMBEDDING_PROVIDER,
            model=CANONICAL_EMBEDDING_MODEL,
            revision=CANONICAL_EMBEDDING_REVISION,
            dimension=CANONICAL_EMBEDDING_DIMENSION,
        )


def expected_vector_identity_from_embedding_configuration(
    configuration: VpiEmbeddingConfiguration,
    *,
    revision: str,
) -> ExpectedVectorIdentity:
    model = configuration.model
    if model is None:
        raise ValueError("embedding model is required")
    if not revision.strip():
        raise ValueError("revision must be non-empty")
    return ExpectedVectorIdentity(
        provider=configuration.provider,
        model=model,
        revision=revision.strip(),
        dimension=configuration.expected_dimension,
    )


@dataclass(frozen=True, slots=True)
class QdrantBootstrapConfiguration:
    integration: QdrantIntegrationConfig
    logical_collection_name: str
    expected_vector_identity: ExpectedVectorIdentity
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE
    vector_transport_tolerance: float = VECTOR_TRANSPORT_FLOAT32_TOLERANCE
    uses_named_dense_vector: bool = False
    dense_vector_channel_name: str = "dense"

    @classmethod
    def from_env(
        cls,
        *,
        logical_collection_name: str = DEFAULT_LOGICAL_COLLECTION_NAME,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
        expected_vector_identity: ExpectedVectorIdentity | None = None,
    ) -> QdrantBootstrapConfiguration:
        if not logical_collection_name.strip():
            raise ValueError("logical_collection_name must be non-empty")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be > 0")
        integration = QdrantIntegrationConfig.from_env(
            collection_name=logical_collection_name,
            enable_sparse_vectors=False,
            metric=CANONICAL_DISTANCE_METRIC,
        )
        return cls(
            integration=integration,
            logical_collection_name=logical_collection_name,
            expected_vector_identity=expected_vector_identity or ExpectedVectorIdentity.canonical_vpi(),
            upsert_batch_size=upsert_batch_size,
        )

    def __repr__(self) -> str:
        return (
            "QdrantBootstrapConfiguration("
            f"logical_collection_name={self.logical_collection_name!r}, "
            f"tenant_id={self.integration.tenant_id!r}, "
            f"upsert_batch_size={self.upsert_batch_size}, "
            f"expected_dimension={self.expected_vector_identity.dimension})"
        )
