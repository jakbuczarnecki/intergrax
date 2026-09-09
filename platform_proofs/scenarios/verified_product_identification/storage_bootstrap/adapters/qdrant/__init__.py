"""Qdrant provider adapter for VPI vector storage bootstrap."""

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.adapter import (
    QdrantVectorStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    CANONICAL_EMBEDDING_DIMENSION,
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_EMBEDDING_PROVIDER,
    CANONICAL_EMBEDDING_REVISION,
    QdrantBootstrapConfiguration,
)

__all__ = (
    "CANONICAL_EMBEDDING_DIMENSION",
    "CANONICAL_EMBEDDING_MODEL",
    "CANONICAL_EMBEDDING_PROVIDER",
    "CANONICAL_EMBEDDING_REVISION",
    "QdrantBootstrapConfiguration",
    "QdrantVectorStorageAdapter",
)
