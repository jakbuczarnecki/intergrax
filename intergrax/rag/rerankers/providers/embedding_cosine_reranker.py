# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from functools import lru_cache
from collections.abc import Sequence
from typing import List, Optional

import numpy as np
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
    RerankerNormalizationMode,
    validate_candidates,
    validate_limit,
)


class EmbeddingCosineReranker(BaseReranker):

    def __init__(
        self,
        embedding_manager: BaseEmbeddingManager,
        *,
        normalization: Optional[RerankerNormalizationMode] = RerankerNormalizationMode.MINMAX,
        fusion_alpha: float = 0.5,
        use_score_fusion: bool = True,
        cache_query_embeddings: bool = True,
        doc_batch_size: int = 256,
    ) -> None:

        self._em = embedding_manager
        self._normalization = normalization
        self._fusion_alpha = fusion_alpha
        self._use_score_fusion = use_score_fusion
        self._doc_batch_size = doc_batch_size
        self._cache_query_embeddings = cache_query_embeddings
        self._embed_query: np.ndarray = None        

    
    @classmethod
    def name(self) -> str:
        return "embedding_cosine"
    

    def _ensure_embed_query(self):
        if self._embed_query is None:
            if self._cache_query_embeddings:
                self._embed_query = lru_cache(maxsize=256)(self._embed_query_uncached)
            else:
                self._embed_query = self._embed_query_uncached


    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        candidates = validate_candidates(candidates)
        validate_limit(limit)
        if not candidates:
            return ()
        
        self._ensure_embed_query()

        normalized = candidates

        if query is None or not query.strip():
            return self._finalize_without_query(normalized, limit)

        texts: List[str] = []
        carriers: List[RerankerCandidate] = []

        for c in normalized:
            txt = c.document.content.strip()
            if not txt:
                continue
            texts.append(txt)
            carriers.append(c)

        if not texts:
            return []

        q_vec = self._embed_query(query)

        doc_vecs = self._embed_texts_batched(texts)

        q = self._l2_norm(q_vec.reshape(1, -1))
        M = self._l2_norm(doc_vecs)

        sims = (q @ M.T).reshape(-1)

        rr_vals: List[float] = []
        orig_vals: List[Optional[float]] = []

        for c, sim in zip(carriers, sims):
            rr_vals.append(float(sim))
            orig_vals.append(c.original_score)

        rr_norm = self._normalize_batch(rr_vals)
        orig_norm = self._normalize_batch(orig_vals)

        scored: list[tuple[RerankerCandidate, float, Optional[float]]] = []

        for idx, candidate in enumerate(carriers):

            rr = rr_norm[idx]
            on = orig_norm[idx]

            fusion: Optional[float] = None

            if self._use_score_fusion:
                if on is not None:
                    fusion = self._fusion_alpha * on + (1 - self._fusion_alpha) * rr
                else:
                    fusion = rr

            scored.append((candidate, rr, fusion))

        scored.sort(
            key=lambda item: item[2] if self._use_score_fusion else item[1],
            reverse=True,
        )
        selected = scored[:limit] if limit is not None else scored
        return tuple(
            RerankerResult(
                candidate=candidate,
                rerank_score=rerank_score,
                fusion_score=fusion_score,
                rank=rank,
            )
            for rank, (candidate, rerank_score, fusion_score) in enumerate(selected)
        )


    def _finalize_without_query(
        self,
        candidates: Sequence[RerankerCandidate],
        limit: int | None,
    ) -> Sequence[RerankerResult]:
        selected = candidates[:limit] if limit is not None else candidates
        return tuple(
            RerankerResult(
                candidate=candidate,
                rerank_score=candidate.original_score,
                fusion_score=(
                    candidate.original_score if self._use_score_fusion else None
                ),
                rank=rank,
            )
            for rank, candidate in enumerate(selected)
        )


    def _embed_query_uncached(self, text: str) -> np.ndarray:
        return np.asarray(self._em.embed_one(text), dtype="float32")


    def _embed_texts_batched(self, texts: List[str]) -> np.ndarray:

        try:
            return np.asarray(self._em.embed_texts(texts), dtype="float32")
        except Exception:

            vecs: List[np.ndarray] = []

            for t in texts:
                vecs.append(self._embed_query_uncached(t))

            return np.vstack(vecs)


    @staticmethod
    def _l2_norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:

        n = np.linalg.norm(X, axis=1, keepdims=True) + eps
        return X / n


    def _normalize_batch(
        self,
        values: List[Optional[float]],
    ) -> List[Optional[float]]:

        xs = [v for v in values if v is not None]

        if not xs:
            return [None for _ in values]

        mode = self._normalization

        if mode is None:
            return values

        arr = np.asarray(xs, dtype=np.float32)

        if mode == RerankerNormalizationMode.MINMAX:

            mn = float(arr.min())
            mx = float(arr.max())

            if abs(mx - mn) < 1e-12:
                norm = [0.5] * len(xs)
            else:
                norm = ((arr - mn) / (mx - mn)).tolist()

        elif mode == RerankerNormalizationMode.ZSCORE:

            mean = float(arr.mean())
            std = float(arr.std())

            if std < 1e-12:
                norm = [0.0] * len(xs)
            else:
                z = (arr - mean) / std
                zmin = float(z.min())
                zmax = float(z.max())

                if abs(zmax - zmin) < 1e-12:
                    norm = [0.5] * len(xs)
                else:
                    norm = ((z - zmin) / (zmax - zmin)).tolist()

        else:
            norm = xs

        out: List[Optional[float]] = []
        idx = 0

        for v in values:
            if v is None:
                out.append(None)
            else:
                out.append(float(norm[idx]))
                idx += 1

        return out