# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import List, Dict, Optional, Any
import logging

from langchain_core.documents import Document
from .vectorstore_manager import VectorstoreManager
from .embedding_manager import EmbeddingManager

logger = logging.getLogger("intergrax.dual_retriever")


class DualRetriever:
    """
    Dual retriever: first query TOC (sections), then fetch local chunks from the same section/source.
    Works with intergraxVectorstoreManager (uses .query, NOT .search).
    """

    def __init__(
        self,
        vs_chunks: VectorstoreManager,
        vs_toc: Optional[VectorstoreManager] = None,
        *,
        embed_manager: Optional[EmbeddingManager] = None,
        k_chunks: int = 30,
        k_toc: int = 8,
        verbose: bool = False,
    ):
        self.vs_chunks = vs_chunks
        self.vs_toc = vs_toc
        self.em = embed_manager  # if not provided, we will raise on first use
        self.k_chunks = int(k_chunks)
        self.k_toc = int(k_toc)
        self.verbose = verbose
        self.log = logger.getChild("retrieve")
        if self.verbose:
            self.log.setLevel(logging.INFO)

    # --- helpers --------------------------------------------------------

    def _ensure_em(self) -> EmbeddingManager:
        if self.em is None:
            raise RuntimeError("intergraxDualRetriever needs an EmbeddingManager (embed_manager=...)")
        return self.em

    def _normalize_hits(self, res: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        intergraxVectorstoreManager.query() returns:
          {"ids":[[...]], "scores":[[...]], "metadatas":[[...]], "documents":[[...]]}
        (for Chroma: 'scores' are similarity in [0..1], because the manager converts distance->1-distance)
        """
        ids_b        = res.get("ids", [[]])
        scores_b     = res.get("scores", [[]])
        metadatas_b  = res.get("metadatas", [[]])
        documents_b  = res.get("documents", [[]])

        ids       = ids_b[0] if ids_b else []
        scores    = scores_b[0] if scores_b else []
        metadatas = metadatas_b[0] if metadatas_b else []
        documents = documents_b[0] if documents_b else []

        n = min(len(ids), len(scores), len(metadatas))
        if isinstance(documents, list) and documents:
            n = min(n, len(documents))

        hits: List[Dict[str, Any]] = []
        for i in range(n):
            md = dict(metadatas[i] or {})
            txt = (documents[i] if (isinstance(documents, list) and i < len(documents)) else md.get("text", "")) or ""
            hits.append({
                "id": ids[i],
                "content": str(txt),
                "metadata": md,
                "similarity_score": float(scores[i]),
                "distance": None,   # in Chroma converted; for other providers we use similarity_score anyway
            })
        # sort descending by similarity
        hits.sort(key=lambda h: h.get("similarity_score", 0.0), reverse=True)
        return hits

    def _query_vs(
        self,
        vs: VectorstoreManager,
        query_text: str,
        *,
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        em = self._ensure_em()
        q_vec = em.embed_one(query_text)  # 1xD (ndarray or list)
        res = vs.query(query_embeddings=q_vec, top_k=top_k, where=where, include_embeddings=False)
        return self._normalize_hits(res)

    def _merge_where_with_parent(
        self,
        where: Optional[Dict[str, Any]],
        parent_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Merge user's `where` filter with a required `parent_id` constraint.

        Supports both:
        - flat dict (legacy) e.g. {"tenant": "x"}
        - Chroma normalized AND form: {"$and":[{...},{...}]}

        Returns None if filters conflict.
        """
        parent_id = str(parent_id)

        # No user filter -> just parent constraint
        if where is None:
            return {"parent_id": {"$eq": parent_id}}

        # Chroma normalized form
        if isinstance(where, dict) and "$and" in where and isinstance(where.get("$and"), list):
            and_list = list(where["$and"])

            # Detect conflicting parent_id if already present in AND
            for cond in and_list:
                if not isinstance(cond, dict):
                    continue
                if "parent_id" in cond:
                    existing = cond.get("parent_id")
                    # existing may be {"$eq": "..."} or plain value
                    if isinstance(existing, dict) and "$eq" in existing:
                        if str(existing["$eq"]) != parent_id:
                            return None
                    elif existing is not None and str(existing) != parent_id:
                        return None

            # Append parent constraint
            and_list.append({"parent_id": {"$eq": parent_id}})
            return {"$and": and_list}

        # Flat dict form
        if isinstance(where, dict):
            # If parent_id already exists and conflicts -> None
            if "parent_id" in where:
                existing = where.get("parent_id")
                if existing is not None and str(existing) != parent_id:
                    return None

            merged = dict(where)
            merged["parent_id"] = parent_id
            return merged

        # Unexpected type -> safest fallback: ignore user filter and enforce parent
        return {"parent_id": {"$eq": parent_id}}


    def _expand_by_toc(self, question: str, where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns local CHUNKS hits expanded based on the sections matched in TOC."""
        if self.vs_toc is None:
            return []

        toc_hits = self._query_vs(self.vs_toc, question, top_k=self.k_toc, where=where)
        self.log.info("[DualRetriever] TOC hits: %d", len(toc_hits))

        expanded: List[Dict[str, Any]] = []
        for h in toc_hits:
            md = h.get("metadata", {}) or {}
            parent = md.get("parent_id")
            if not parent:
                # No reliable join key between TOC and CHUNKS => cannot expand
                if self.verbose:
                    self.log.info("[DualRetriever] TOC hit without parent_id -> skipping TOC expansion for this item.")
                continue

            where_local = self._merge_where_with_parent(where, parent)
            if where_local is None:
                # filter collision (e.g., where['parent_id'] differs from matched parent)
                continue

            local = self._query_vs(self.vs_chunks, question, top_k=self.k_chunks, where=where_local)
            expanded.extend(local)

        self.log.info("[DualRetriever] Expanded via TOC to %d local hits", len(expanded))
        return expanded

    # --- public ---------------------------------------------------------

    def retrieve(self, question: str, *, top_k: int = 40, where: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        1) Fetch base hits from CHUNKS (max(self.k_chunks, top_k))
        2) Expand context via TOC (if available) and search locally by parent_id (propagates `where`)
        3) Merge, dedupe and sort by similarity, then trim to top_k
        """
        self.log.info("[DualRetriever] Query: '%s' (top_k=%d)", question, top_k)

        base_k = max(int(top_k), self.k_chunks)
        base_hits = self._query_vs(self.vs_chunks, question, top_k=base_k, where=where)
        self.log.info("[DualRetriever] Base CHUNKS hits: %d", len(base_hits))

        toc_expanded = self._expand_by_toc(question, where)

        # merge + dedupe
        def _key(h: Dict[str, Any]):
            m = h.get("metadata", {}) or {}
            return (h.get("id"), m.get("chunk_id"), m.get("parent_id"))

        seen, merged = set(), []
        for h in (toc_expanded + base_hits):
            k = _key(h)
            if k in seen:
                continue
            seen.add(k)
            merged.append(h)

        merged.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        out = merged[: int(top_k)]
        self.log.info("[DualRetriever] Total merged: %d", len(out))
        return out
