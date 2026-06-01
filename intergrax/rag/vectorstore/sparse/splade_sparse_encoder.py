# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Optional SPLADE sparse encoder (requires ``fastembed``)."""

from __future__ import annotations

import os
from typing import List, Optional

from intergrax.rag.vectorstore.sparse.bm25_sparse_encoder import SparseVector

_DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


class SpladeSparseEncoder:
    """
    Learned sparse encoder using ``fastembed.SparseTextEmbedding``.

    Install: ``pip install fastembed`` (optional — not required for CI default path).
    """

    def __init__(self, *, model_name: Optional[str] = None) -> None:
        self._model_name = (
            model_name
            or os.getenv("INTERGRAX_RAG_SPLADE_MODEL", "").strip()
            or _DEFAULT_MODEL
        )
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as exc:
            raise ImportError(
                "SPLADE sparse encoder requires fastembed: pip install fastembed"
            ) from exc
        self._model = SparseTextEmbedding(model_name=self._model_name)
        return self._model

    def encode(self, text: str) -> SparseVector:
        if not (text or "").strip():
            return SparseVector(indices=[], values=[])
        model = self._ensure_model()
        embed = getattr(model, "embed")
        raw = next(embed([text]))
        indices: List[int] = []
        values: List[float] = []
        for token_id, weight in raw.as_dict().items():
            indices.append(int(token_id))
            values.append(float(weight))
        if not indices:
            return SparseVector(indices=[], values=[])
        pairs = sorted(zip(indices, values), key=lambda p: p[0])
        return SparseVector(
            indices=[p[0] for p in pairs],
            values=[p[1] for p in pairs],
        )
