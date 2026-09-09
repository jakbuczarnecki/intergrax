"""Unit tests for vector index identity compatibility gate."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_compatibility import (
    VectorIndexCompatibilityGate,
    verify_vector_index_compatible,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_expectations import (
    expected_vector_index_identity_from_bootstrap_vector_identity,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_metadata import (
    VectorIndexPersistedMetadata,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_index_runtime_identity import (
    ExpectedVectorIndexRuntimeIdentity,
    ResolvedVectorIndexRuntimeIdentity,
    VectorIndexIdentityResolution,
    VectorIndexIdentityResolutionStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    ExpectedVectorIdentity,
)

pytestmark = pytest.mark.unit

_TARGET = VectorIndexIdentity(logical_name="vpi-product-embeddings", tenant_id="default")
_EMBEDDING = ExpectedVectorIdentity(
    provider="hf",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimension=4,
)


def _expected(*, content_identity: str | None = None) -> ExpectedVectorIndexRuntimeIdentity:
    return expected_vector_index_identity_from_bootstrap_vector_identity(
        target=_TARGET,
        embedding_identity=_EMBEDDING,
        content_identity=content_identity,
    )


def _resolved(
    *,
    provider: str = "hf",
    model: str = "BAAI/bge-m3",
    revision: str = "5617a9f61b028005a4858fdac845db406aefb181",
    dimension: int = 4,
    metric: str = "cosine",
    content_identity: str | None = None,
    target: VectorIndexIdentity = _TARGET,
) -> ResolvedVectorIndexRuntimeIdentity:
    return ResolvedVectorIndexRuntimeIdentity(
        target=target,
        exists=True,
        reachable=True,
        provider=provider,
        model=model,
        revision=revision,
        dimension=dimension,
        metric=metric,
        content_identity=content_identity,
    )


@dataclass(slots=True)
class _FakeVectorIndexIdentityResolver:
    resolution: VectorIndexIdentityResolution

    def resolve(self, expected: ExpectedVectorIndexRuntimeIdentity) -> VectorIndexIdentityResolution:
        return self.resolution


def test_exact_identity_match_passes() -> None:
    expected = _expected()
    assert verify_vector_index_compatible(expected=expected, actual=_resolved()) is None


def test_provider_mismatch_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(),
        actual=_resolved(provider="openai"),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_model_mismatch_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(),
        actual=_resolved(model="other/model"),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_revision_mismatch_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(),
        actual=_resolved(revision="other-revision"),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_dimension_mismatch_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(),
        actual=_resolved(dimension=768),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_metric_mismatch_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(),
        actual=_resolved(metric="dot"),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_content_identity_mismatch_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(content_identity="content-a"),
        actual=_resolved(content_identity="content-b"),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_missing_required_content_identity_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(content_identity="content-a"),
        actual=_resolved(content_identity=None),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_missing_embedding_metadata_fails_closed() -> None:
    failure = verify_vector_index_compatible(
        expected=_expected(),
        actual=_resolved(revision=None),
    )
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_provider_unavailable_maps_to_unavailable() -> None:
    gate = VectorIndexCompatibilityGate(
        expected=_expected(),
        resolver=_FakeVectorIndexIdentityResolver(
            resolution=VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.PROVIDER_UNAVAILABLE,
                identity=None,
            )
        ),
    )
    failure = gate.ensure_compatible()
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.UNAVAILABLE


def test_missing_collection_maps_to_invalid_query() -> None:
    gate = VectorIndexCompatibilityGate(
        expected=_expected(),
        resolver=_FakeVectorIndexIdentityResolver(
            resolution=VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.INDEX_MISSING,
                identity=None,
            )
        ),
    )
    failure = gate.ensure_compatible()
    assert failure is not None
    assert failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_success_is_cached() -> None:
    calls = 0

    @dataclass(slots=True)
    class _CountingResolver:
        def resolve(self, expected: ExpectedVectorIndexRuntimeIdentity) -> VectorIndexIdentityResolution:
            nonlocal calls
            calls += 1
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.RESOLVED,
                identity=_resolved(),
            )

    gate = VectorIndexCompatibilityGate(expected=_expected(), resolver=_CountingResolver())
    assert gate.ensure_compatible() is None
    assert gate.ensure_compatible() is None
    assert calls == 1


def test_transient_failure_is_not_cached() -> None:
    calls = 0

    @dataclass(slots=True)
    class _FlakyResolver:
        def resolve(self, expected: ExpectedVectorIndexRuntimeIdentity) -> VectorIndexIdentityResolution:
            nonlocal calls
            calls += 1
            if calls == 1:
                return VectorIndexIdentityResolution(
                    status=VectorIndexIdentityResolutionStatus.PROVIDER_UNAVAILABLE,
                    identity=None,
                )
            return VectorIndexIdentityResolution(
                status=VectorIndexIdentityResolutionStatus.RESOLVED,
                identity=_resolved(),
            )

    gate = VectorIndexCompatibilityGate(expected=_expected(), resolver=_FlakyResolver())
    assert gate.ensure_compatible() is not None
    assert gate.ensure_compatible() is None
    assert calls == 2


def test_bootstrap_metadata_roundtrip() -> None:
    metadata = VectorIndexPersistedMetadata(
        target=_TARGET,
        content_identity="abc123",
        provider=_EMBEDDING.provider,
        model=_EMBEDDING.model,
        revision=_EMBEDDING.revision,
        dimension=_EMBEDDING.dimension,
    )
    payload = metadata.to_provider_payload(
        embedding_provider_key="embedding_provider",
        embedding_model_key="embedding_model",
        embedding_revision_key="embedding_revision",
        embedding_dimension_key="embedding_dimension",
    )
    restored = VectorIndexPersistedMetadata.from_provider_payload(
        payload,
        embedding_provider_key="embedding_provider",
        embedding_model_key="embedding_model",
        embedding_revision_key="embedding_revision",
        embedding_dimension_key="embedding_dimension",
    )
    assert restored == metadata
