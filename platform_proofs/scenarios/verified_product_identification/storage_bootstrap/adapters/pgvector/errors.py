"""PgVector bootstrap adapter errors — translated at port boundary."""

from __future__ import annotations


class PgVectorBootstrapAdapterError(Exception):
    """Base adapter error for pgvector storage bootstrap."""


class PgVectorBootstrapConfigurationError(PgVectorBootstrapAdapterError):
    """Invalid adapter configuration or unmapped logical vector target."""


class PgVectorBootstrapSchemaError(PgVectorBootstrapAdapterError):
    """Schema preparation, extension, or compatibility failure."""


class PgVectorBootstrapVectorValidationError(PgVectorBootstrapAdapterError):
    """Malformed or incompatible vector record before provider write."""


class PgVectorBootstrapIdentityConflictError(PgVectorBootstrapAdapterError):
    """Canonical point identity already exists with incompatible immutable content."""


class PgVectorBootstrapOperationError(PgVectorBootstrapAdapterError):
    """Bounded pgvector operation failure without leaking provider details."""
