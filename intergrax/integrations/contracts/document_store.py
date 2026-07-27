# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Document / wide-column store integration contract (§7.1.2, Phase M.6 P2)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Normalized document row scoped by partition key."""

    partition_key: str
    row_key: str
    data: Mapping[str, Any] = Field(default_factory=dict)
    ttl_seconds: Optional[int] = None


class DocumentQueryResult(BaseModel):
    documents: Sequence[DocumentRecord] = Field(default_factory=list)
    total: int = 0


@runtime_checkable
class DocumentStore(Protocol):
    """
    Backend-agnostic document store facade.

    Implementations: cassandra, mongodb, dynamodb, …
    """

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        """Fetch a single document or return ``None`` when missing."""

    def put(self, document: DocumentRecord) -> None:
        """Insert or upsert a document."""

    def delete(self, partition_key: str, row_key: str) -> None:
        """Remove a document."""

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        """List documents within a partition, optionally filtered by row-key prefix."""

    def close(self) -> None:
        """Release resources."""


@runtime_checkable
class ConditionalDocumentStore(DocumentStore, Protocol):
    """
    Optional single-record conditional write capability.

    Compare partition_key, row_key and data only — not ttl_seconds.
    Normal conflicts return False; they are not errors.
    """

    def put_if_absent(
        self,
        document: DocumentRecord,
    ) -> bool:
        """Atomically insert when missing; return False if the key already exists."""

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        """Atomically replace when current record matches expected keys+data."""

    def delete_if_match(
        self,
        *,
        expected: DocumentRecord,
    ) -> bool:
        """Atomically delete when current record matches expected keys+data."""
