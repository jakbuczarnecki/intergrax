# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned reference document-store materialization (EBH-3-R2)."""

from __future__ import annotations

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore


def create_reference_in_memory_document_store(
    *,
    cursor_secret: bytes | None = None,
) -> DocumentStore:
    """Return the canonical in-process reference ``DocumentStore`` implementation."""
    if cursor_secret is None:
        return InMemoryDocumentStore()
    return InMemoryDocumentStore(cursor_secret=cursor_secret)


__all__ = ["create_reference_in_memory_document_store"]
