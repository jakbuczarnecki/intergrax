# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

import uuid
from collections.abc import Mapping, Sequence
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreRecord
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
    def _doc_texts(docs: Sequence[KnowledgeDocument]) -> List[str]:
        return [d.content for d in docs]

    @staticmethod
    def _doc_payloads(
        docs: Sequence[KnowledgeDocument],
        base: Mapping[str, JsonValue] | None = None,
    ) -> List[dict[str, JsonValue]]:
        base = base or {}
        out: List[dict[str, JsonValue]] = []
        for d in docs:
            md = dict(base)
            md.update(dict(d.metadata))
            out.append(md)
        return out

    @staticmethod
    def _record_documents(records: Sequence[VectorStoreRecord]) -> List[KnowledgeDocument]:
        return [record.document for record in records]

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