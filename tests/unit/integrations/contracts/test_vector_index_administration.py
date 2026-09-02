"""Unit tests for provider-neutral vector index administration contracts."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.vector_index_administration import (
    DenseVectorChannelSpec,
    SparseLexicalChannelSpec,
    VectorIndexCompatibilityError,
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorIndexPrepareOutcome,
    VectorIndexPrepareResult,
    VectorIndexSpec,
    VectorSearchCapability,
    validate_spec_against_description,
)

pytestmark = pytest.mark.unit


def _identity() -> VectorIndexIdentity:
    return VectorIndexIdentity(logical_name="catalog", tenant_id="default")


def _spec(*, dimension: int = 1024, sparse: bool = True) -> VectorIndexSpec:
    sparse_spec = SparseLexicalChannelSpec(channel_name="sparse") if sparse else None
    required = {VectorSearchCapability.DENSE}
    if sparse:
        required.add(VectorSearchCapability.SPARSE_LEXICAL)
    return VectorIndexSpec(
        identity=_identity(),
        dense=DenseVectorChannelSpec(
            channel_name="dense",
            dimension=dimension,
            metric="cosine",
        ),
        required_capabilities=frozenset(required),
        sparse_lexical=sparse_spec,
    )


def test_vector_index_spec_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        VectorIndexSpec(
            identity=_identity(),
            dense=DenseVectorChannelSpec(channel_name="dense", dimension=0, metric="cosine"),
            required_capabilities=frozenset({VectorSearchCapability.DENSE}),
        )


def test_vector_index_spec_requires_sparse_spec_when_capability_requested() -> None:
    with pytest.raises(ValueError, match="sparse_lexical"):
        VectorIndexSpec(
            identity=_identity(),
            dense=DenseVectorChannelSpec(channel_name="dense", dimension=8, metric="cosine"),
            required_capabilities=frozenset(
                {VectorSearchCapability.DENSE, VectorSearchCapability.SPARSE_LEXICAL}
            ),
            sparse_lexical=None,
        )


def test_validate_spec_against_description_accepts_compatible_state() -> None:
    description = VectorIndexDescription(
        identity=_identity(),
        exists=True,
        reachable=True,
        point_count=10,
        dense_dimension=1024,
        present_capabilities=frozenset(
            {VectorSearchCapability.DENSE, VectorSearchCapability.SPARSE_LEXICAL}
        ),
        dense_channel_name="dense",
        sparse_lexical_channel_name="sparse",
    )
    validate_spec_against_description(_spec(dimension=1024), description)


def test_validate_spec_against_description_rejects_dimension_mismatch() -> None:
    description = VectorIndexDescription(
        identity=_identity(),
        exists=True,
        reachable=True,
        point_count=10,
        dense_dimension=512,
        present_capabilities=frozenset(
            {VectorSearchCapability.DENSE, VectorSearchCapability.SPARSE_LEXICAL}
        ),
        dense_channel_name="dense",
        sparse_lexical_channel_name="sparse",
    )
    with pytest.raises(VectorIndexCompatibilityError, match="dimension"):
        validate_spec_against_description(_spec(dimension=1024), description)


def test_validate_spec_against_description_rejects_missing_sparse_capability() -> None:
    description = VectorIndexDescription(
        identity=_identity(),
        exists=True,
        reachable=True,
        point_count=0,
        dense_dimension=1024,
        present_capabilities=frozenset({VectorSearchCapability.DENSE}),
        dense_channel_name="dense",
        sparse_lexical_channel_name=None,
    )
    with pytest.raises(VectorIndexCompatibilityError, match="sparse_lexical"):
        validate_spec_against_description(_spec(dimension=1024, sparse=True), description)


def test_prepare_result_dtos_are_immutable() -> None:
    description = VectorIndexDescription(
        identity=_identity(),
        exists=True,
        reachable=True,
        point_count=0,
        dense_dimension=8,
        present_capabilities=frozenset({VectorSearchCapability.DENSE}),
        dense_channel_name="dense",
        sparse_lexical_channel_name=None,
    )
    result = VectorIndexPrepareResult(
        outcome=VectorIndexPrepareOutcome.CREATED,
        description=description,
    )
    with pytest.raises(AttributeError):
        result.outcome = VectorIndexPrepareOutcome.ALREADY_COMPATIBLE  # type: ignore[misc]
