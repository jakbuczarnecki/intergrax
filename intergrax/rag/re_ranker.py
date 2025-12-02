# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import logging
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger("intergrax.reranker")

# Input types: hits from the retriever (dict) or raw LangChain Documents
Hit = Dict[str, Any]
Candidates = Union[List[Hit], List[Document]]

@dataclass(frozen=True)
class ReRankerConfig:
    """Config for intergraxReRanker."""
    # Score fusion: combine original retriever score with rerank score.
    # fusion_alpha=0 → only rerank score, 1.0 → only original retriever score
    use_score_fusion: bool = True
    fusion_alpha: float = 0.5  # final = alpha*orig_sim + (1-alpha)*rerank_score
    # Normalization before fusion (stabilization): "minmax" | "zscore" | None
    normalize: Optional[str] = "minmax"
    # Batch sizes
    query_batch_size: int = 512     # for potential multi-query scenarios
    doc_batch_size: int = 256
    # Cache query embeddings (by text hash). Good trade-off for repeated questions.
    cache_query_embeddings: bool = True
    # How to read text from hits
    # default fields: 'content' or metadata['text']
    hit_text_key: str = "content"
    meta_text_key: str = "text"
    # Key of original score from retriever (if you want fusion)
    hit_orig_score_key: str = "similarity_score"

class ReRanker:
    """
    Fast, scalable cosine re-ranker over candidate chunks.
    - Accepts hits from intergraxRagRetriever (dict) OR raw LangChain Documents.
    - Embeds texts in batches using intergraxEmbeddingManager.
    - Optional score fusion with original retriever similarity.
    - Preserves schema of hits; only adds:
        - 'rerank_score': float in [0..1] (cosine on L2-normalized vectors)
        - 'fusion_score': float (if use_score_fusion=True)
        - 'rank_reranked': final integer rank (1-based)
    """

    def __init__(self,
                 embedding_manager,  # intergraxEmbeddingManager
                 config: Optional[ReRankerConfig] = None,
                 *,
                 verbose: bool = False) -> None:
        self.em = embedding_manager
        self.cfg = config or ReRankerConfig()
        self.verbose = verbose
        self.log = logger.getChild("core")
        if self.verbose:
            self.log.setLevel(logging.INFO)

        # ensure alpha in [0,1]
        if self.cfg.fusion_alpha is not None:
            a = float(self.cfg.fusion_alpha)
            if not (0.0 <= a <= 1.0):
                raise ValueError("fusion_alpha must be in [0,1]")

        # lightweight in-memory cache for query embeddings
        if self.cfg.cache_query_embeddings:
            # wrap embed_one with LRU by text
            self._embed_query_cached = lru_cache(maxsize=256)(self._embed_query_no_cache)
        else:
            self._embed_query_cached = self._embed_query_no_cache

    # ---------- public API ----------

    def __call__(self, *args, **kwargs) -> List[Hit]:
        """
        Supports:
          a) __call__(query=..., candidates=..., top_k=...)
          b) __call__(query, candidates, top_k)
          c) __call__(candidates)
        """
        # a) keywords
        if ("candidates" in kwargs) or ("query" in kwargs) or ("top_k" in kwargs):
            query: Optional[str] = kwargs.get("query")
            candidates: Optional[Candidates] = kwargs.get("candidates")
            top_k: Optional[int] = kwargs.get("top_k")
            if candidates is None:
                raise TypeError("intergraxReRanker.__call__ requires 'candidates' when using keyword arguments.")
            return self.rerank_candidates(query=query, candidates=candidates, rerank_k=top_k)

        # b/c) positional
        if len(args) == 1:
            # only candidates
            candidates = args[0]
            return self.rerank_candidates(query=None, candidates=candidates, rerank_k=None)
        elif len(args) == 2:
            # (query, candidates)
            query, candidates = args
            return self.rerank_candidates(query=query, candidates=candidates, rerank_k=None)
        elif len(args) >= 3:
            # (query, candidates, top_k)
            query, candidates, top_k = args[0], args[1], args[2]
            return self.rerank_candidates(query=query, candidates=candidates, rerank_k=top_k)

        raise TypeError("intergraxReRanker.__call__ expected (candidates) or (query, candidates[, top_k]) or keywords.")

    def rerank_candidates(self,
                          query: Optional[str],
                          candidates: Candidates,
                          *,
                          rerank_k: Optional[int] = None) -> List[Hit]:
        """
        Re-rank a given candidate list by cosine similarity to the query.
        Returns list of hit dicts (schema preserved + 'rerank_score', optional 'fusion_score').
        If query is None/empty → no re-ranking (optionally sort by original similarity_score).
        """
        if not candidates:
            return []

        # No query provided → return as-is (or sort by original score)
        if query is None or not str(query).strip():
            hits_out = self._ensure_hit_dicts(candidates)
            # if original scores exist, sort by them
            if any(isinstance(h.get(self.cfg.hit_orig_score_key), (int, float)) for h in hits_out):
                hits_out.sort(key=lambda x: float(x.get(self.cfg.hit_orig_score_key, 0.0)), reverse=True)
            if rerank_k is not None and rerank_k > 0:
                hits_out = hits_out[: int(rerank_k)]
            # set rank_reranked for consistency
            for r, h in enumerate(hits_out, start=1):
                h["rank_reranked"] = r
            return hits_out

        # 1) Prepare raw texts and "carry" structs referencing original objects
        texts: List[str] = []
        carriers: List[Tuple[int, Union[Hit, Document]]] = []

        if isinstance(candidates[0], Document):
            for i, d in enumerate(candidates):  # type: ignore[arg-type]
                if not isinstance(d, Document):
                    continue
                t = (d.page_content or "").strip()
                if not t:
                    continue
                texts.append(t)
                carriers.append((i, d))
        else:
            # hits (dict)
            for i, h in enumerate(candidates):  # type: ignore[arg-type]
                if not isinstance(h, dict):
                    continue
                # content → text → metadata['text']
                t = (h.get(self.cfg.hit_text_key)
                     or h.get("text")
                     or h.get("page_content")
                     or h.get("metadata", {}).get(self.cfg.meta_text_key, ""))
                t = str(t).strip()
                if not t:
                    continue
                texts.append(t)
                carriers.append((i, h))

        if not texts:
            return self._ensure_hit_dicts(candidates)

        # 2) Embed query + documents (batched), L2-normalize
        q_vec = self._embed_query_cached(query)
        D = self._embed_texts_batched(texts, batch_size=self.cfg.doc_batch_size)
        q = self._l2_norm(q_vec.reshape(1, -1))
        M = self._l2_norm(D)

        # 3) Cosine similarities: q (1 x d) · M.T (d x n) → (1 x n)
        sims = (q @ M.T).astype("float32").reshape(-1)  # in [-1..1], with L2 norm typically [0..1]

        # 4) Build result hit dicts and compute rerank scores
        hits_out: List[Hit] = []
        for (orig_idx, obj), sim in zip(carriers, sims):
            hit = self._to_hit(obj)
            hit["rerank_score"] = float(sim)
            # Optional fusion with original retriever score
            if self.cfg.use_score_fusion:
                orig = hit.get(self.cfg.hit_orig_score_key)
                if isinstance(orig, (int, float)):
                    s_orig = float(orig)
                    s_rr = float(sim)
                    s_orig_n = self._normalize_scalar(s_orig, self.cfg.normalize)
                    s_rr_n = self._normalize_scalar(s_rr, self.cfg.normalize)
                    alpha = float(self.cfg.fusion_alpha)
                    hit["fusion_score"] = alpha * s_orig_n + (1.0 - alpha) * s_rr_n
                else:
                    hit["fusion_score"] = float(sim)  # fallback: same as rerank_score
            hits_out.append(hit)

        # 5) Final sort by 'fusion_score' (if enabled) else by 'rerank_score'
        key = ("fusion_score" if self.cfg.use_score_fusion else "rerank_score")
        hits_out.sort(key=lambda x: float(x.get(key, 0.0)), reverse=True)

        # 6) Cap, add 'rank_reranked'
        if rerank_k is not None and rerank_k > 0:
            hits_out = hits_out[: int(rerank_k)]
        for r, h in enumerate(hits_out, start=1):
            h["rank_reranked"] = r

        if self.verbose and hits_out:
            top = hits_out[0]
            dbg = top.get(key, top.get("rerank_score", 0.0))
            self.log.info("[intergraxReRanker] Top1 %s=%.4f  (text≈ %s...)", key, float(dbg),
                          str(top.get("content",""))[:120].replace("\n"," "))

        return hits_out

    # ---------- convenience wrapper over retriever ----------

    def rerank_via_retriever(self,
                             query: str,
                             *,
                             base_retriever,                 # intergraxRagRetriever
                             retriever_k: int = 30,
                             rerank_k: int = 10,
                             score_threshold: float = 0.0,
                             where: Optional[Dict[str, Any]] = None) -> List[Hit]:
        """
        Convenience: recall with retriever (broad), then re-rank here.
        """
        if self.verbose:
            self.log.info("[intergraxReRanker] Recall: top_k=%d, thr=%.4f", retriever_k, score_threshold)
        base_hits = base_retriever.retrieve(
            question=query,
            top_k=retriever_k,
            score_threshold=score_threshold,
            where=where,
            include_embeddings=False,   # not needed: we embed here
            use_mmr=True,               # recommendation: diversify before re-ranking
            prefetch_factor=5,
        )
        return self.rerank_candidates(query=query, candidates=base_hits, rerank_k=rerank_k)

    # ---------- internals ----------

    def _embed_query_no_cache(self, text: str) -> np.ndarray:
        v = self.em.embed_one(text)
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype="float32")
        return v.astype("float32")

    def _embed_texts_batched(self, texts: List[str], batch_size: int) -> np.ndarray:
        # If you have a fast API that embeds a list of texts, prefer it.
        # Try .embed_texts; if unavailable, fall back to per-item embed_one.
        if hasattr(self.em, "embed_texts"):
            V = self.em.embed_texts(texts)  # type: ignore[attr-defined]
            return np.asarray(V, dtype="float32")
        # fallback: small batches on embed_one
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            part = texts[i:i+batch_size]
            for t in part:
                vecs.append(self._embed_query_cached(t))
        return np.vstack(vecs).astype("float32")

    @staticmethod
    def _l2_norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True) + eps
        return X / n

    def _normalize_scalar(self, x: float, mode: Optional[str]) -> float:
        """Per-value normalization for fusion. We assume score ranges ~[0..1]; provide light stabilization."""
        if mode is None:
            return x
        if mode == "minmax":
            # clamp to [0,1] if slightly out of range due to numerics
            return float(max(0.0, min(1.0, x)))
        if mode == "zscore":
            # without batch stats, center around 0.5 (approximation)
            return float((x - 0.5) / 0.25)  # 0.5→0, [0..1]→[-2..+2]
        return x

    def _to_hit(self, obj: Union[Hit, Document]) -> Hit:
        """Return a hit-dict; preserve input dict; convert Document to hit-like dict."""
        if isinstance(obj, dict):
            return dict(obj)
        # Document → dict
        meta = dict(obj.metadata or {})
        return {
            "id": meta.get("id") or meta.get("doc_id") or meta.get("chunk_id") or "",
            "content": obj.page_content or "",
            "text": obj.page_content or "",
            "page_content": obj.page_content or "",
            "metadata": meta,
            "similarity_score": meta.get("similarity_score"),  # if present
            "distance": meta.get("distance"),                  # if present
            "rank": meta.get("rank"),                          # if present
        }

    def _ensure_hit_dicts(self, candidates: Candidates) -> List[Hit]:
        out: List[Hit] = []
        if not candidates:
            return out
        if isinstance(candidates[0], Document):
            for d in candidates:  # type: ignore[arg-type]
                out.append(self._to_hit(d))
        else:
            for h in candidates:  # type: ignore[arg-type]
                out.append(self._to_hit(h))
        return out
