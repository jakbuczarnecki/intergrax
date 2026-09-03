"""Explicit embedding materialization failure hierarchy."""

from __future__ import annotations


class VpiEmbeddingMaterializationError(Exception):
    """Base error for VPI embedding artifact materialization."""


class ArtifactCompatibilityError(VpiEmbeddingMaterializationError):
    """Existing artifact identity does not match active configuration."""


class ArtifactIntegrityError(VpiEmbeddingMaterializationError):
    """Artifact shard or manifest integrity validation failed."""


class ArtifactWriteError(VpiEmbeddingMaterializationError):
    """Artifact shard or manifest write failed."""


class EmbeddingMaterializationConfigurationError(VpiEmbeddingMaterializationError):
    """Invalid or incomplete materialization configuration."""


class EmbeddingMaterializationDataError(VpiEmbeddingMaterializationError):
    """Dataset parsing or derivation failure during materialization."""


class EmbeddingMaterializationProviderError(VpiEmbeddingMaterializationError):
    """Embedding provider readiness or batch operation failure."""
