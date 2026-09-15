# © Artur Czarnecki. All rights reserved.

"""Projection failure classification for memory lifecycle (provider-agnostic)."""

from __future__ import annotations

from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionFailureCategory,
    MemoryProjectionFailureEvidence,
    MemoryProjectionOperation,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreContractError

__all__ = [
    "classify_memory_projection_failure",
]


def classify_memory_projection_failure(
    *,
    projection_id: str,
    operation: MemoryProjectionOperation,
    exc: BaseException,
) -> MemoryProjectionFailureEvidence:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        category = MemoryProjectionFailureCategory.RETRYABLE
    elif isinstance(exc, VectorStoreContractError):
        category = MemoryProjectionFailureCategory.PERMANENT
    elif isinstance(exc, ValueError):
        category = MemoryProjectionFailureCategory.PERMANENT
    else:
        category = MemoryProjectionFailureCategory.RETRYABLE
    message = exc.__class__.__name__
    return MemoryProjectionFailureEvidence(
        projection_id=projection_id,
        operation=operation,
        category=category,
        message=message,
    )
