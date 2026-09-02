"""Scenario-owned runtime configuration."""

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    VPI_EMBEDDING_ENV_PREFIX,
    VPI_REFERENCE_EMBEDDING_DIMENSION,
    VPI_REFERENCE_EMBEDDING_MODEL,
    VPI_REFERENCE_EMBEDDING_PROVIDER,
    VpiEmbeddingConfiguration,
    VpiEmbeddingDimensionMismatchError,
    VpiIndexEmbeddingIdentity,
    load_vpi_embedding_configuration,
    validate_resolved_provider_dimension,
)

__all__ = [
    "EMBEDDING_CONFIGURATION_VERSION",
    "VPI_EMBEDDING_ENV_PREFIX",
    "VPI_REFERENCE_EMBEDDING_DIMENSION",
    "VPI_REFERENCE_EMBEDDING_MODEL",
    "VPI_REFERENCE_EMBEDDING_PROVIDER",
    "VpiEmbeddingConfiguration",
    "VpiEmbeddingDimensionMismatchError",
    "VpiIndexEmbeddingIdentity",
    "load_vpi_embedding_configuration",
    "validate_resolved_provider_dimension",
]
