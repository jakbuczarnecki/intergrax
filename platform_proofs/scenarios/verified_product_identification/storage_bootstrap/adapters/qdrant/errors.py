"""Qdrant vector bootstrap adapter errors — translated at port boundary."""

from __future__ import annotations


class QdrantBootstrapAdapterError(Exception):
    """Base adapter error for Qdrant vector storage bootstrap."""


class QdrantBootstrapConfigurationError(QdrantBootstrapAdapterError):
    """Invalid adapter configuration or unmapped logical vector target."""


class QdrantBootstrapCollectionError(QdrantBootstrapAdapterError):
    """Collection preparation or compatibility failure."""


class QdrantBootstrapVectorValidationError(QdrantBootstrapAdapterError):
    """Malformed or incompatible vector record before provider write."""


class QdrantBootstrapIdentityConflictError(QdrantBootstrapAdapterError):
    """Canonical point identity already exists with incompatible immutable content."""


class QdrantBootstrapOperationError(QdrantBootstrapAdapterError):
    """Bounded Qdrant operation failure without leaking provider details."""
