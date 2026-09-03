# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral hosted document retrieval (managed stores + provider query API)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationError


class ManagedRetrievalConfigurationError(IntegrationError):
    """Provider binding or credentials are missing or invalid."""


class ManagedRetrievalResourceNotFoundError(IntegrationError):
    """Managed store or attached resource does not exist."""


class ManagedRetrievalUploadError(IntegrationError):
    """Document upload or attach failed."""


class ManagedRetrievalProcessingTimeoutError(IntegrationError):
    """Provider did not finish processing an uploaded document in time."""


class ManagedRetrievalQueryError(IntegrationError):
    """Hosted retrieval query failed."""


@dataclass(frozen=True, slots=True)
class ManagedRetrievalUploadResult:
    """Normalized folder upload outcome."""

    uploaded_names: tuple[str, ...]
    failed_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManagedRetrievalQueryRequest:
    """Neutral query payload — business instructions resolved before the adapter."""

    store_id: str
    question: str
    model: str
    instructions: str
    max_results: int
    score_threshold: float


@runtime_checkable
class ManagedRetrievalBackend(Protocol):
    """
    Hosted managed retrieval facade (provider-managed files + indexes + query API).

    Distinct from raw ``VectorStore`` vector upsert/query and from ``DocumentStore``
    application persistence.
    """

    def ensure_store_exists(self, store_id: str) -> None:
        """Verify the managed store resource exists and is reachable."""

    def list_attached_file_ids(self, store_id: str) -> Sequence[str]:
        """List provider file/resource ids attached to the managed store."""

    def upload_folder(
        self,
        store_id: str,
        folder: str | Path,
        *,
        patterns: Sequence[str],
    ) -> ManagedRetrievalUploadResult:
        """Upload local files and attach them to the managed store."""

    def clear_store(self, store_id: str) -> int:
        """Detach and delete provider file resources; return successful deletion count."""

    def query(self, request: ManagedRetrievalQueryRequest) -> str:
        """Run a hosted retrieval query and return normalized answer text."""


__all__ = [
    "ManagedRetrievalBackend",
    "ManagedRetrievalConfigurationError",
    "ManagedRetrievalProcessingTimeoutError",
    "ManagedRetrievalQueryError",
    "ManagedRetrievalQueryRequest",
    "ManagedRetrievalResourceNotFoundError",
    "ManagedRetrievalUploadError",
    "ManagedRetrievalUploadResult",
]
