# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.vectorstore.sparse.bm25_sparse_encoder import encode_sparse_bm25

pytestmark = pytest.mark.unit


def test_encode_sparse_bm25_non_empty() -> None:
    vec = encode_sparse_bm25("Intergrax harness RAG pipeline")
    assert vec.indices
    assert vec.values
    assert len(vec.indices) == len(vec.values)


def test_encode_sparse_bm25_empty() -> None:
    vec = encode_sparse_bm25("")
    assert vec.indices == []
    assert vec.values == []
