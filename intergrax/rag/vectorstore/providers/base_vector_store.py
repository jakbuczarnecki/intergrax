# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from langchain_core.documents import Document

from intergrax.rag.vectorstore.contracts.vector_store import VectorStore


class BaseVectorStore(VectorStore):
    """
    Shared helper base class for vector store providers.

    Contains provider-agnostic helper utilities extracted
    from VectorstoreManager without behavioral changes.
    """

    @staticmethod
    def _to_list_of_lists(
        emb: Union[NDArray[np.float32], Sequence[Sequence[float]]]
    ) -> List[List[float]]:
        if isinstance(emb, np.ndarray):
            if emb.ndim == 1:
                emb = np.expand_dims(emb, axis=0)
            return emb.astype(np.float32).tolist()
        return [list(map(float, v)) for v in emb]

    @staticmethod
    def _doc_texts(docs: Sequence[Document]) -> List[str]:
        return [d.page_content or "" for d in docs]

    @staticmethod
    def _doc_payloads(
        docs: Sequence[Document],
        base: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        base = base or {}
        out: List[Dict[str, Any]] = []
        for d in docs:
            md = dict(base)
            md.update(dict(d.metadata))
            out.append(md)
        return out

    @staticmethod
    def _make_ids(n: int, prefix: str = "doc") -> List[str]:
        return [f"{prefix}_{uuid.uuid4().hex[:8]}_{i}" for i in range(n)]

    def _ensure_dim_consistency(
        self,
        batch: Sequence[Sequence[float]],
    ) -> None:
        if not batch:
            return
        if attribute_access.optional(self, "_dim", None) is None:
            self._dim = len(batch[0])
        else:
            bad = [
                i for i, v in enumerate(batch)
                if len(v) != self._dim
            ]
            if bad:
                raise ValueError(
                    f"Inconsistent embedding dimension in batch at positions {bad[:5]} "
                    f"(expected {self._dim})."
                )