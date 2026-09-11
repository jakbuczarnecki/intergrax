"""Typed operator-layer errors for VPI full storage load."""

from __future__ import annotations


class StorageLoadOperatorError(Exception):
    """Base operator-layer error."""


class StorageLoadOperatorPreconditionError(StorageLoadOperatorError):
    """Operator configuration or artifact preconditions failed before load."""


class StorageLoadOperatorLockError(StorageLoadOperatorError):
    """Single-writer operator lock could not be acquired."""


class StorageLoadOperatorEvidenceError(StorageLoadOperatorError):
    """Operator evidence directory is unusable."""


class StorageLoadOperatorUsageError(StorageLoadOperatorError):
    """CLI usage or mode combination is invalid."""
