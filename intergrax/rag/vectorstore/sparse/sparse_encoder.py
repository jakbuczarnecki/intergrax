# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Pluggable sparse encoders for native hybrid vector stores."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from intergrax.rag.vectorstore.sparse.bm25_sparse_encoder import SparseVector, encode_sparse_bm25


@runtime_checkable
class SparseEncoder(Protocol):
    def encode(self, text: str) -> SparseVector: ...


class Bm25HashSparseEncoder:
    """Default hash-BM25 sparse vectors (no external model)."""

    def encode(self, text: str) -> SparseVector:
        return encode_sparse_bm25(text)


def resolve_sparse_encoder(
    mode: str | None = None,
    *,
    model_name: str | None = None,
) -> SparseEncoder:
    """
    Resolve sparse encoder from mode or ``INTERGRAX_RAG_SPARSE_ENCODER``.

    Modes:
    - ``bm25_hash`` (default)
    - ``splade`` — learned sparse via optional ``fastembed`` package
    """
    resolved = (mode or os.getenv("INTERGRAX_RAG_SPARSE_ENCODER", "bm25_hash")).strip().lower()
    if resolved in ("bm25", "bm25_hash", "hash"):
        return Bm25HashSparseEncoder()
    if resolved == "splade":
        from intergrax.rag.vectorstore.sparse.splade_sparse_encoder import SpladeSparseEncoder

        return SpladeSparseEncoder(model_name=model_name)
    return Bm25HashSparseEncoder()


def encode_sparse(text: str, *, encoder: SparseEncoder | None = None) -> SparseVector:
    enc = encoder or resolve_sparse_encoder()
    return enc.encode(text)
