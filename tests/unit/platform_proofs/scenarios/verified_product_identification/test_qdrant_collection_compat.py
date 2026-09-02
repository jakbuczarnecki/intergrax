"""Unit tests for Qdrant collection compatibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant.collection_compat import (
    collection_dense_dimension,
    collection_has_sparse_channel,
)

pytestmark = pytest.mark.unit


def _collection_info(*, dense_size: int, sparse_names: tuple[str, ...]) -> SimpleNamespace:
    dense = SimpleNamespace(size=dense_size)
    sparse_vectors = {name: SimpleNamespace() for name in sparse_names}
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": dense},
                sparse_vectors=sparse_vectors,
            )
        )
    )


def test_collection_dense_dimension_matches() -> None:
    info = _collection_info(dense_size=1024, sparse_names=("sparse",))
    assert collection_dense_dimension(info, dense_vector_name="dense") == 1024


def test_collection_sparse_channel_present() -> None:
    info = _collection_info(dense_size=1024, sparse_names=("sparse",))
    assert collection_has_sparse_channel(info, sparse_vector_name="sparse") is True


def test_dense_only_collection_has_no_sparse_channel() -> None:
    info = _collection_info(dense_size=1024, sparse_names=())
    assert collection_has_sparse_channel(info, sparse_vector_name="sparse") is False
