"""Fail-closed vector index compatibility verification for VPI retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_runtime_identity import (
    ExpectedVectorIndexRuntimeIdentity,
    ResolvedVectorIndexRuntimeIdentity,
    VectorIndexIdentityResolution,
    VectorIndexIdentityResolutionStatus,
)


@runtime_checkable
class VectorIndexIdentityResolver(Protocol):
    def resolve(
        self,
        expected: ExpectedVectorIndexRuntimeIdentity,
    ) -> VectorIndexIdentityResolution: ...


def verify_vector_index_compatible(
    *,
    expected: ExpectedVectorIndexRuntimeIdentity,
    actual: ResolvedVectorIndexRuntimeIdentity,
) -> CatalogSearchFailure | None:
    if not actual.exists:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index does not exist",
        )
    if actual.target != expected.target:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index target mismatch",
        )
    if actual.dimension is None:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index dimension unavailable",
        )
    if actual.dimension != expected.dimension:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index dimension mismatch",
        )
    if actual.metric is None:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index distance metric unavailable",
        )
    if actual.metric != expected.metric:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index distance metric mismatch",
        )
    if actual.provider is None or actual.model is None or actual.revision is None:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index embedding identity unavailable",
        )
    if actual.provider != expected.provider:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index embedding provider mismatch",
        )
    if actual.model != expected.model:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index embedding model mismatch",
        )
    if actual.revision != expected.revision:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index embedding revision mismatch",
        )
    if expected.content_identity is not None:
        if actual.content_identity is None:
            return CatalogSearchFailure(
                kind=CatalogSearchFailureKind.INVALID_QUERY,
                message="vector index content identity unavailable",
            )
        if actual.content_identity != expected.content_identity:
            return CatalogSearchFailure(
                kind=CatalogSearchFailureKind.INVALID_QUERY,
                message="vector index content identity mismatch",
            )
    return None


def map_identity_resolution_failure(
    resolution: VectorIndexIdentityResolution,
) -> CatalogSearchFailure:
    if resolution.status is VectorIndexIdentityResolutionStatus.PROVIDER_UNAVAILABLE:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="vector index administration unavailable",
        )
    if resolution.status is VectorIndexIdentityResolutionStatus.INDEX_MISSING:
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector index does not exist",
        )
    return CatalogSearchFailure(
        kind=CatalogSearchFailureKind.INVALID_QUERY,
        message="vector index identity metadata unavailable",
    )


@dataclass(slots=True)
class VectorIndexCompatibilityGate:
    """Lazy one-time compatibility gate with success-only caching."""

    expected: ExpectedVectorIndexRuntimeIdentity
    resolver: VectorIndexIdentityResolver
    _compatible: bool = False

    def ensure_compatible(self) -> CatalogSearchFailure | None:
        if self._compatible:
            return None
        resolution = self.resolver.resolve(self.expected)
        if resolution.status is not VectorIndexIdentityResolutionStatus.RESOLVED:
            return map_identity_resolution_failure(resolution)
        if resolution.identity is None:
            return CatalogSearchFailure(
                kind=CatalogSearchFailureKind.INVALID_QUERY,
                message="vector index identity unavailable",
            )
        failure = verify_vector_index_compatible(
            expected=self.expected,
            actual=resolution.identity,
        )
        if failure is not None:
            return failure
        self._compatible = True
        return None


__all__ = [
    "VectorIndexCompatibilityGate",
    "VectorIndexIdentityResolver",
    "map_identity_resolution_failure",
    "verify_vector_index_compatible",
]
