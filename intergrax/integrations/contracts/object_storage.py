# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Object / blob storage integration contract (§7.1.2, Phase M.6 P2)."""

from __future__ import annotations

from typing import Literal, Mapping, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

PresignedUrlMethod = Literal["GET", "PUT"]


class StoredObject(BaseModel):
    """Normalized object payload returned by ``ObjectStorage.get``."""

    key: str
    body: bytes
    content_type: str = "application/octet-stream"
    metadata: Mapping[str, str] = Field(default_factory=dict)
    size_bytes: int = 0


@runtime_checkable
class ObjectStorage(Protocol):
    """
    Backend-agnostic blob storage facade.

    Implementations: s3, azure_blob, gcs, filesystem, …
    """

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Upload or overwrite an object at ``key``."""

    def get(self, key: str) -> Optional[StoredObject]:
        """Fetch object bytes or return ``None`` when missing."""

    def delete(self, key: str) -> None:
        """Remove an object."""

    def presigned_url(
        self,
        key: str,
        *,
        expires_in_seconds: int = 3600,
        method: PresignedUrlMethod = "GET",
    ) -> str:
        """Return a time-limited URL for direct client upload/download."""

    def close(self) -> None:
        """Release resources."""
