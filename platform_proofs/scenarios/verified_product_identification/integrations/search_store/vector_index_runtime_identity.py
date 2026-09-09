"""Typed vector index runtime identity contracts for VPI retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.rag.vectorstore.config.vector_config import Metric


@dataclass(frozen=True, slots=True)
class ExpectedVectorIndexRuntimeIdentity:
    """Expected index identity derived from VPI configuration and bootstrap contracts."""

    target: VectorIndexIdentity
    provider: str
    model: str
    revision: str
    dimension: int
    metric: Metric
    content_identity: str | None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.revision.strip():
            raise ValueError("revision must be non-empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be > 0")
        if self.content_identity is not None and not self.content_identity.strip():
            raise ValueError("content_identity must be non-empty when set")


@dataclass(frozen=True, slots=True)
class ResolvedVectorIndexRuntimeIdentity:
    """Authoritative index identity resolved from provider state."""

    target: VectorIndexIdentity
    exists: bool
    reachable: bool
    provider: str | None
    model: str | None
    revision: str | None
    dimension: int | None
    metric: Metric | None
    content_identity: str | None


class VectorIndexIdentityResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INDEX_MISSING = "index_missing"
    METADATA_UNAVAILABLE = "metadata_unavailable"


@dataclass(frozen=True, slots=True)
class VectorIndexIdentityResolution:
    status: VectorIndexIdentityResolutionStatus
    identity: ResolvedVectorIndexRuntimeIdentity | None

    def __post_init__(self) -> None:
        if self.status is VectorIndexIdentityResolutionStatus.RESOLVED:
            if self.identity is None:
                raise ValueError("resolved identity is required when status is RESOLVED")
            return
        if self.identity is not None:
            raise ValueError("identity must be omitted unless status is RESOLVED")


__all__ = [
    "ExpectedVectorIndexRuntimeIdentity",
    "ResolvedVectorIndexRuntimeIdentity",
    "VectorIndexIdentityResolution",
    "VectorIndexIdentityResolutionStatus",
]
