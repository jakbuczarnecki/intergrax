"""Typed Qdrant collection compatibility helpers for VPI bootstrap."""

from __future__ import annotations

from collections.abc import Mapping


def is_collection_not_found(exc: BaseException) -> bool:
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
    except ImportError:
        return False
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, UnexpectedResponse) and current.status_code == 404:
            return True
        current = current.__cause__
    return "404" in str(exc)


def collection_dense_dimension(collection_info, *, dense_vector_name: str) -> int | None:
    vectors = collection_info.config.params.vectors
    if vectors is None:
        return None
    if isinstance(vectors, Mapping):
        dense = vectors.get(dense_vector_name)
        if dense is not None:
            return int(dense.size)
        if len(vectors) == 1:
            only = next(iter(vectors.values()))
            return int(only.size)
        return None
    return int(vectors.size)


def collection_has_sparse_channel(collection_info, *, sparse_vector_name: str) -> bool:
    sparse_vectors = collection_info.config.params.sparse_vectors
    if sparse_vectors is None:
        return False
    if isinstance(sparse_vectors, Mapping):
        return sparse_vector_name in sparse_vectors
    return False
