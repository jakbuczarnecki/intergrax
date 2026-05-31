# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.vectorstore.sparse.bm25_sparse_encoder import encode_sparse_bm25
from intergrax.rag.vectorstore.sparse.sparse_encoder import (
    Bm25HashSparseEncoder,
    resolve_sparse_encoder,
)

pytestmark = pytest.mark.unit


def test_resolve_sparse_encoder_bm25_default() -> None:
    enc = resolve_sparse_encoder("bm25_hash")
    assert isinstance(enc, Bm25HashSparseEncoder)
    vec = enc.encode("Intergrax harness RAG pipeline")
    direct = encode_sparse_bm25("Intergrax harness RAG pipeline")
    assert vec.indices == direct.indices
    assert vec.values == direct.values


def test_splade_encoder_requires_fastembed() -> None:
    pytest.importorskip("fastembed")
    from intergrax.rag.vectorstore.sparse.splade_sparse_encoder import SpladeSparseEncoder

    enc = SpladeSparseEncoder()
    vec = enc.encode("Intergrax retrieval metrics")
    assert vec.indices
    assert len(vec.indices) == len(vec.values)
