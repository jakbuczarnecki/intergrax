# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from langchain_core.documents import Document
import numpy as np

from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.contracts.vector_store import VectorStoreHit
from intergrax.logging import IntergraxLogging

logger = IntergraxLogging.get_logger(__name__, component="rag")


class VectorstoreManager(BaseVectorstoreManager):
    """
    Thin delegation layer for VectorStore providers.

    This class no longer implements any backend logic.
    It delegates all operations to the injected VectorStore instance.
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store


    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        docs = list(documents)
        if base_metadata:
            for d in docs:
                d.metadata = {**(d.metadata or {}), **base_metadata}

        if len(docs) != len(embeddings):
            raise ValueError(
                "VectorstoreManager.add_documents: "
                "documents and embeddings length mismatch "
                f"({len(docs)} vs {len(embeddings)})"
            )

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        dim = len(embeddings[0])
        for i, vec in enumerate(embeddings):

            if len(vec) != dim:
                raise ValueError(
                    "Embedding dimension mismatch "
                    f"at index {i}: {len(vec)} != {dim}"
                )
            
            for v in vec:
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(
                        f"Invalid embedding value at index {i}"
                    )

        self._store.add_documents(
            documents=docs,
            embeddings=embeddings,
            ids=ids,
        )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        return self._store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_embeddings=include_embeddings,
        )

    def delete(self, ids: Sequence[str]) -> None:
        self._store.delete(ids)

    def count(self) -> int:
        return self._store.count()

    def list_collections(self) -> list[str]:
        return list(self._store.list_collections())