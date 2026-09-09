"""PgVector provider adapter for VPI vector storage bootstrap."""

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.adapter import (
    PgVectorStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.configuration import (
    CANONICAL_EMBEDDING_DIMENSION,
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_EMBEDDING_PROVIDER,
    CANONICAL_EMBEDDING_REVISION,
    PgVectorBootstrapConfiguration,
)

__all__ = (
    "CANONICAL_EMBEDDING_DIMENSION",
    "CANONICAL_EMBEDDING_MODEL",
    "CANONICAL_EMBEDDING_PROVIDER",
    "CANONICAL_EMBEDDING_REVISION",
    "PgVectorBootstrapConfiguration",
    "PgVectorStorageAdapter",
)
