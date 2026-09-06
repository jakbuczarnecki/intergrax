"""Typed data-pack errors."""

from __future__ import annotations


class VpiDataPackError(RuntimeError):
    """Base error for VPI universal data pack operations."""


class VpiDataPackValidationError(VpiDataPackError):
    """Raised when artifact validation fails."""


class VpiDataPackBuildError(VpiDataPackError):
    """Raised when build cannot complete."""


class VpiDataPackIntegrityError(VpiDataPackError):
    """Raised when checksum or shard integrity checks fail."""


class VpiDataPackCompatibilityError(VpiDataPackError):
    """Raised when manifest or identity compatibility checks fail."""


class VpiDataPackFormatError(VpiDataPackError):
    """Raised when manifest or index JSON is malformed or uses invalid types."""


class EmbeddingModelIdentityError(VpiDataPackError):
    """Raised when embedding model revision or fingerprint cannot be resolved."""
