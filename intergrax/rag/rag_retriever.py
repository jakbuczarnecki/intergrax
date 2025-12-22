# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np

from .vectorstore_manager import VectorstoreManager
from .embedding_manager import EmbeddingManager

# Optional reranker function type: accepts and returns a list of hits
RerankerFn = Callable[..., List[Dict[str, Any]]]


class RagRetriever:
    """
    Scalable, provider-agnostic RAG retriever for intergrax.

    ### Key Features
    - Normalizes `where` filters for Chroma (flat dict → $and/$eq)
    - Normalizes query vector shape (1D/2D → [[D]])
    - Unified similarity scoring:
        * Chroma → converts distance to similarity = 1 - distance
        * Others → uses raw similarity as returned
    - Deduplication by ID + per-parent result limiting (diversification)
    - Optional MMR diversification when embeddings are returned
    - Batch retrieval for multiple queries
    - Optional reranker hook (e.g., cross-encoder, re-ranking model)

    ### Returned structure (list of dicts)
    Each hit includes:
    ```
    {
      "id": str,
      "content": str,
      "metadata": dict,
      "similarity_score": float,
      "distance": float,  # only for Chroma; omitted for other providers
      "rank": int,
      "embedding": Optional[List[float]]
    }
    ```
    """

    def __init__(
        self,
        vector_store: VectorstoreManager,
        embedding_manager: EmbeddingManager,
        *,
        verbose: bool = False,
        default_max_per_parent: Optional[int] = 2,
        chroma_auto_where_normalize: bool = True,
    ):
        self.vs = vector_store
        self.em = embedding_manager
        self.verbose = verbose
        self.default_max_per_parent = default_max_per_parent
        self.chroma_auto_where_normalize = chroma_auto_where_normalize

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_where(self, where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Converts flat dict filters into a proper Chroma `$and/$eq` structure.

        Example:
            {"tenant": "intergrax", "corpus": "v1"}
        becomes:
            {"$and": [{"tenant": {"$eq": "intergrax"}}, {"corpus": {"$eq": "v1"}}]}
        """
        if not where:
            return None
        if self.vs.cfg.provider != "chroma":
            return where
        if not self.chroma_auto_where_normalize:
            return where
        if any(str(k).startswith("$") for k in where.keys()):
            return where
        return {"$and": [{k: {"$eq": v}} for k, v in where.items()]}

    @staticmethod
    def _as_query_batch(vec: Any) -> List[List[float]]:
        """Ensure the embedding is formatted as [[D]] regardless of input shape."""
        if isinstance(vec, np.ndarray):
            if vec.ndim == 1:
                return [vec.astype("float32").tolist()]
            if vec.ndim == 2:
                return vec.astype("float32").tolist()
        if isinstance(vec, list):
            if vec and isinstance(vec[0], (list, np.ndarray)):
                return [list(map(float, row)) for row in vec]
            return [list(map(float, vec))]
        return [[float(vec)]]

    @staticmethod
    def _scores_to_similarity(scores: Sequence[float], provider: str) -> List[float]:
        """Convert provider-specific scores into similarity values in [0..1]."""
        if provider == "chroma":
            return [min(1.0, max(0.0, 1.0 - float(d))) for d in scores]
        return [float(s) for s in scores]

    @staticmethod
    def _mmr(
        query_vec: np.ndarray,
        cand_vecs: np.ndarray,
        k: int,
        lambda_mult: float = 0.5,
    ) -> List[int]:
        """
        Maximal Marginal Relevance diversification.

        Args:
            query_vec: (D,) array - query embedding.
            cand_vecs: (N, D) array - candidate embeddings.
            k: number of items to select.
            lambda_mult: trade-off between relevance (to query) and diversity (to other items).

        Returns:
            List of selected indices (indices in cand_vecs).
        """
        if k <= 0:
            return []

        # Ensure shapes + dtype
        q_in = np.asarray(query_vec, dtype="float32").reshape(-1)          # (D,)
        C_in = np.asarray(cand_vecs, dtype="float32")
        if C_in.ndim != 2 or C_in.shape[0] == 0 or C_in.shape[1] == 0:
            return []

        k = min(int(k), int(C_in.shape[0]))

        # Clamp lambda to [0, 1]
        if not isinstance(lambda_mult, (int, float)):
            lambda_mult = 0.5
        lambda_mult = float(lambda_mult)
        if lambda_mult < 0.0:
            lambda_mult = 0.0
        if lambda_mult > 1.0:
            lambda_mult = 1.0

        def _unit(x: np.ndarray) -> np.ndarray:
            """
            L2-normalize on the last axis.
            Always returns float32 ndarray.
            """
            x = np.asarray(x, dtype="float32")
            n = np.linalg.norm(x, axis=-1, keepdims=True).astype("float32") + np.float32(1e-12)
            return (x / n).astype("float32")

        q = _unit(q_in)     # (D,)
        C = _unit(C_in)     # (N, D)

        # Similarity to query for each candidate: (N,)
        sim_q = (C @ q.reshape(-1, 1)).ravel()

        selected: List[int] = []
        candidates: List[int] = list(range(int(C.shape[0])))

        while len(selected) < k and candidates:
            if not selected:
                # Pick highest relevance first
                best_pos = int(np.argmax(sim_q[candidates]))
                selected.append(candidates.pop(best_pos))
                continue

            # Penalize redundancy relative to selected items
            S = C[selected]               # (m, D)
            cand_mat = C[candidates]      # (c, D)
            sim_div = cand_mat @ S.T      # (c, m)
            max_div = np.max(sim_div, axis=1)  # (c,)

            mmr_score = (lambda_mult * sim_q[candidates]) - ((1.0 - lambda_mult) * max_div)
            best_pos = int(np.argmax(mmr_score))
            selected.append(candidates.pop(best_pos))

        return selected

    

    def _batch_embed_contents(self, texts: List[str]) -> np.ndarray:
        """
        Robustly embed a list of texts using intergraxEmbeddingManager.
        Handles different return signatures:
        - returns ndarray directly
        - or (ndarray, anything)
        - or list[list[float]]
        - or falls back to per-item embed_one
        """
        # Preferred: vectorized method exists
        if callable(self.em.embed_texts):
            try:
                res = self.em.embed_texts(texts)  # may return ndarray OR (ndarray, ...)
                # If it looks like a tuple/list, try to take first element as ndarray
                if isinstance(res, (tuple, list)) and len(res) > 0:
                    vecs = res[0]
                else:
                    vecs = res
                # Normalize to ndarray
                if isinstance(vecs, np.ndarray):
                    return vecs.astype("float32")
                if isinstance(vecs, list) and vecs and isinstance(vecs[0], (list, tuple, np.ndarray)):
                    return np.array(vecs, dtype="float32")
            except Exception:
                # Fall through to per-item
                pass

        # Fallback: per-item
        arrs: List[np.ndarray] = []
        for t in texts:
            v = self.em.embed_one(t)  # expected 1xD
            # ensure 1D row
            v = np.array(v, dtype="float32").reshape(-1)
            arrs.append(v)
        return np.vstack(arrs).astype("float32")



    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        *,
        score_threshold: float = 0.0,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
        max_per_parent: Optional[int] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        reranker: Optional[RerankerFn] = None,
        prefetch_factor: int = 5,   # fetch a wider candidate pool from the vector DB
    ) -> List[Dict[str, Any]]:
        # Guard: empty store
        try:            
            if int(self.vs.count() or 0) == 0:
                if self.verbose:
                    print("[intergraxRagRetriever] Vector store is empty.")
                return []
        except Exception:
            # If count() is unsupported, ignore and continue
            pass

        used_mmr = False
        used_reranker = False

        # 1) Embed query → [[D]]
        q_vec_1d = self.em.embed_one(question)
        Q = self._as_query_batch(q_vec_1d)

        # 2) Normalize filters (Chroma: flat → {$and:[{k:{$eq:v}}...]})
        norm_where = self._normalize_where(where)

        # 3) Prefetch from the vector DB (with filter fallbacks for Chroma)
        raw_top_k = max(int(top_k), 1)
        prefetch_k = max(raw_top_k * int(prefetch_factor or 1), raw_top_k)

        def _do_query(_where):
            return self.vs.query(
                query_embeddings=Q,
                top_k=prefetch_k,
                where=_where,
                include_embeddings=include_embeddings,
            )

        # Try normalized → flat → none
        res = _do_query(norm_where)
        if (not res.get("ids") or not res["ids"][0]) and where and self.vs.cfg.provider == "chroma":
            if self.verbose:
                print("[Retriever] No hits with normalized where → retrying with flat where…")
            res = _do_query(where)
        if (not res.get("ids") or not res["ids"][0]) and self.vs.cfg.provider == "chroma":
            if self.verbose:
                print("[Retriever] Still no hits → retrying with no filter (diagnostic)…")
            res = _do_query(None)

        provider = self.vs.cfg.provider
        ids_b = res.get("ids", [[]])
        scores_b = res.get("scores", [[]])  # Chroma: distances; others: similarity
        metas_b = res.get("metadatas", [[]])
        docs_b = res.get("documents", [[]])
        embs_b = res.get("embeddings", [[]]) if include_embeddings else [[]]

        ids = ids_b[0] if ids_b else []
        raw_scores = scores_b[0] if scores_b else []
        metadatas = metas_b[0] if metas_b else []
        documents = docs_b[0] if docs_b else []
        emb_vecs = embs_b[0] if include_embeddings and embs_b else []

        if not ids:
            if self.verbose:
                print("[intergraxRagRetriever] No results from vector store.")
            return []

        # 4) Normalize scores → similarity in [0,1]
        sims = self._scores_to_similarity(raw_scores, provider)

        # Align lengths (do NOT clamp by emb_vecs length — embeddings may be absent)
        n = min(len(ids), len(metadatas), len(sims))
        docs_present = isinstance(documents, list) and any(documents)
        if docs_present:
            n = min(n, len(documents))

        if self.verbose:
            print(
                f"[Retriever] raw candidates: {n}, "
                f"min_sim={min(sims[:n] or [0]):.3f}, max_sim={max(sims[:n] or [0]):.3f}, "
                f"threshold={score_threshold:.3f}, prefetch_k={prefetch_k}"
            )

        # 5) Build candidates WITHOUT early thresholding
        cands: List[Dict[str, Any]] = []
        for i in range(n):
            meta = dict(metadatas[i] or {})
            content = (
                documents[i]
                if (docs_present and i < len(documents) and documents[i])
                else str(meta.get("text", ""))
            )
            item = {
                "id": str(ids[i]),
                "content": content,
                "metadata": meta,
                "similarity_score": float(sims[i]),
            }

            if provider == "chroma":
                # raw_scores are distances for chroma
                try:
                    item["distance"] = float(raw_scores[i])
                except Exception:
                    item["distance"] = float(1.0 - sims[i])

            # Embeddings from provider (optional)
            if include_embeddings and i < len(emb_vecs) and emb_vecs[i] is not None:
                item["embedding"] = emb_vecs[i]
            cands.append(item)

        if not cands:
            return []

        # 6) Deduplicate by ID
        seen_ids = set()
        uniq: List[Dict[str, Any]] = []
        for it in cands:
            _id = it.get("id")
            if _id in seen_ids:
                continue
            seen_ids.add(_id)
            uniq.append(it)

        # Keep a defensive snapshot BEFORE MMR/rerank so we can fall back safely
        uniq_before_mmr = list(uniq)

        # 7) Optional MMR: ensure we HAVE embeddings; if not, compute them on the fly
        if use_mmr:
            have_embs = include_embeddings and any("embedding" in x for x in uniq)

            if not have_embs:
                try:
                    texts = [x["content"] for x in uniq]
                    C_list = self._batch_embed_contents(texts)  # helper assumed to exist
                    for i, x in enumerate(uniq):
                        x["embedding"] = C_list[i]
                    have_embs = True
                    if self.verbose:
                        print("[intergraxRagRetriever] Computed candidate embeddings on-the-fly for MMR.")
                except Exception as e:
                    have_embs = False
                    if self.verbose:
                        print(f"[intergraxRagRetriever] Could not compute embeddings on-the-fly; skipping MMR. Err: {e}")

            if have_embs:
                try:
                    # Safe coerce embeddings to 1D float32 and keep mapping to uniq indices
                    emb_items: List[np.ndarray] = []
                    emb_indices: List[int] = []

                    for idx, it in enumerate(uniq):
                        emb = it.get("embedding")
                        if emb is None:
                            continue
                        try:
                            v = np.asarray(emb, dtype="float32").reshape(-1)
                            emb_items.append(v)
                            emb_indices.append(idx)
                        except Exception:
                            continue

                    if len(emb_items) >= 2:
                        q = np.asarray(q_vec_1d, dtype="float32").reshape(-1)
                        C = np.vstack(emb_items).astype("float32")

                        k_for_mmr = min(len(emb_items), prefetch_k)
                        order_local = self._mmr(q, C, k=k_for_mmr, lambda_mult=mmr_lambda)

                        if order_local:
                            selected_idxs = [emb_indices[i] for i in order_local]
                            selected_set = set(selected_idxs)

                            # Keep the MMR-selected items first, then append the remaining candidates
                            uniq = [uniq[i] for i in selected_idxs] + [it for j, it in enumerate(uniq) if j not in selected_set]
                            used_mmr = True
                    else:
                        if self.verbose:
                            print("[intergraxRagRetriever] MMR skipped: not enough valid embeddings.")
                except Exception as e:
                    if self.verbose:
                        print(f"[intergraxRagRetriever] MMR error ignored: {e}")
                    uniq = uniq_before_mmr  # safe fallback


            # If something went wrong and we ended up empty, revert safely
            if not uniq:
                uniq = uniq_before_mmr

        # 8) Now apply threshold
        if score_threshold > 0.0:
            uniq = [it for it in uniq if it.get("similarity_score", 0.0) >= score_threshold]

        if not uniq:
            # Defensive fallback: return top_k by raw similarity from the original candidates
            uniq = sorted(cands, key=lambda x: x.get("similarity_score", 0.0), reverse=True)[:raw_top_k]

       
        # Snapshot before rerank (for safe fallback)
        uniq_before_rerank = list(uniq)

        # 9a) Optional reranker (e.g., cross-encoder)
        if callable(reranker):
            try:
                out = None
                try:
                    # prefer signature compatible with intergrax ReRanker
                    out = reranker(candidates=uniq, query=question, top_k=raw_top_k)  # type: ignore[call-arg]
                except TypeError:
                    try:
                        # fallback: rerankers that accept (hits, query=?, top_k=?)
                        out = reranker(uniq, query=question, top_k=raw_top_k)  # type: ignore[call-arg]
                    except TypeError:
                        # legacy: rerankers that only take the list
                        out = reranker(uniq)

                if isinstance(out, list) and out:
                    uniq = out
                    used_reranker = True
                elif isinstance(out, list) and not out:
                    # If reranker returns [], treat as "no-op" and fallback
                    uniq = uniq_before_rerank
                else:
                    # Unexpected return type -> fallback
                    uniq = uniq_before_rerank
            except Exception as e:
                if self.verbose:
                    print(f"[intergraxRagRetriever] Reranker error ignored: {e}")
                uniq = uniq_before_rerank


        # 9a.1) If reranker was used, sort by reranker-provided score before diversification/cap.
        # Prefer fusion_score (if present), otherwise rerank_score.
        if used_reranker:
            def _rerank_key(x: Dict[str, Any]) -> float:
                fs = x.get("fusion_score")
                if isinstance(fs, (int, float)):
                    return float(fs)
                rs = x.get("rerank_score")
                if isinstance(rs, (int, float)):
                    return float(rs)
                return 0.0

            uniq.sort(key=_rerank_key, reverse=True)

        
        # 9b) Per-parent limit AFTER reranker (final diversification)
        limit = self.default_max_per_parent if max_per_parent is None else max_per_parent
        if limit and limit > 0:
            buckets: Dict[str, int] = {}
            diversified: List[Dict[str, Any]] = []
            for it in uniq:
                md = it.get("metadata") or {}
                
                parent = (
                    md.get("parent_id")
                    or md.get("source_path")
                    or md.get("source")
                    or md.get("source_name")
                )

                # If we cannot determine a stable parent, skip per-parent limiting for this item
                if not parent:
                    diversified.append(it)
                    continue

                parent = str(parent)

                cnt = buckets.get(parent, 0)
                if cnt < limit:
                    diversified.append(it)
                    buckets[parent] = cnt + 1
            uniq = diversified or uniq
        
        # 10) Final sort, rank, and cap
        # IMPORTANT: do NOT destroy the order produced by MMR/reranker.
        if not used_mmr and not used_reranker:
            uniq.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

        for r, it in enumerate(uniq, start=1):
            it["rank"] = r

        return uniq[:raw_top_k]


    # ------------------------------------------------------------------
    # Batch retrieval
    # ------------------------------------------------------------------
    def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
        max_per_parent: Optional[int] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        reranker: Optional[RerankerFn] = None,
        prefetch_factor: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run retrieval for multiple queries sequentially (simple loop-based batching).

        Returns:
            List of results per query.
        """
        out: List[List[Dict[str, Any]]] = []
        for q in queries:
            hits = self.retrieve(
                question=q,
                top_k=top_k,
                score_threshold=score_threshold,
                where=where,
                include_embeddings=include_embeddings,
                max_per_parent=max_per_parent,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                reranker=reranker,
                prefetch_factor=prefetch_factor
            )
            out.append(hits)
        return out
