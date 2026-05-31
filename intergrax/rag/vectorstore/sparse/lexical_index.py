# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""In-process BM25-style lexical index for hybrid retrieval (M-RAG.10)."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


class LexicalIndex:
    """Lightweight BM25 index — no external sparse vector SDK required."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._doc_terms: Dict[str, Dict[str, int]] = {}
        self._doc_lens: Dict[str, int] = {}
        self._df: Dict[str, int] = defaultdict(int)
        self._avg_dl: float = 0.0

    def upsert(self, doc_id: str, text: str) -> None:
        terms = tokenize(text)
        if not terms:
            return
        tf: Dict[str, int] = defaultdict(int)
        for t in terms:
            tf[t] += 1
        if doc_id in self._doc_terms:
            for term in self._doc_terms[doc_id]:
                self._df[term] = max(0, self._df[term] - 1)
        self._doc_terms[doc_id] = dict(tf)
        self._doc_lens[doc_id] = len(terms)
        for term in tf:
            self._df[term] += 1
        n = len(self._doc_lens)
        self._avg_dl = sum(self._doc_lens.values()) / n if n else 0.0

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._doc_terms:
            return
        for term in self._doc_terms[doc_id]:
            self._df[term] = max(0, self._df[term] - 1)
        del self._doc_terms[doc_id]
        del self._doc_lens[doc_id]
        n = len(self._doc_lens)
        self._avg_dl = sum(self._doc_lens.values()) / n if n else 0.0

    def search(
        self,
        query_text: str,
        *,
        top_k: int,
        allowed_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        q_terms = tokenize(query_text)
        if not q_terms or not self._doc_terms:
            return []

        n_docs = len(self._doc_terms)
        scores: List[Tuple[str, float]] = []

        for doc_id, tf_map in self._doc_terms.items():
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            dl = self._doc_lens.get(doc_id, 0)
            score = 0.0
            for term in q_terms:
                if term not in tf_map:
                    continue
                df = self._df.get(term, 0)
                if df <= 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_map[term]
                denom = tf + self._k1 * (1.0 - self._b + self._b * dl / max(self._avg_dl, 1.0))
                score += idf * (tf * (self._k1 + 1.0)) / max(denom, 1e-9)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if not scores:
            return []
        max_s = scores[0][1]
        return [(doc_id, s / max_s if max_s > 0 else 0.0) for doc_id, s in scores[:top_k]]
