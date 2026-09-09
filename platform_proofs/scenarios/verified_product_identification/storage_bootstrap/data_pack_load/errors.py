"""Typed storage bootstrap failures for Data Pack load orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BootstrapFailureCategory(str, Enum):
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    RELATIONAL_WRITE_FAILED = "RELATIONAL_WRITE_FAILED"
    VECTOR_WRITE_FAILED = "VECTOR_WRITE_FAILED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    PARTIAL_BATCH = "PARTIAL_BATCH"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    INTEGRITY_FAILED = "INTEGRITY_FAILED"


@dataclass(frozen=True, slots=True)
class BootstrapFailure:
    category: BootstrapFailureCategory
    detail: str
    batch_number: int | None = None
    first_failed_identity: str | None = None


class StorageBootstrapError(Exception):
    """Base error for Data Pack storage bootstrap orchestration."""


class StorageBootstrapPreconditionError(StorageBootstrapError):
    """Invalid request or Data Pack compatibility failure."""


class StorageBootstrapIdentityError(StorageBootstrapError):
    """Cross-artifact identity contract violation."""


class StorageBootstrapWriteError(StorageBootstrapError):
    """Provider write failure during batch load."""


class StorageBootstrapIntegrityError(StorageBootstrapError):
    """Post-write verification or batch integrity failure."""
