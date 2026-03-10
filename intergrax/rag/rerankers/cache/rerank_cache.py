# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Tuple


class RerankCache:
    """
    In-memory cache for reranker score results.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 3600,
    ) -> None:

        self._ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, List[float]]] = {}

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _hash_documents(self, texts: List[str]) -> str:

        joined = "\n".join(texts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _build_key(
        self,
        reranker: str,
        query: str,
        texts: List[str],
    ) -> str:

        q_hash = self._hash_query(query)
        d_hash = self._hash_documents(texts)

        return f"rerank:{reranker}:{q_hash}:{d_hash}"

    def get(
        self,
        *,
        reranker: str,
        query: str,
        texts: List[str],
    ) -> Optional[List[float]]:

        key = self._build_key(reranker, query, texts)

        entry = self._store.get(key)

        if not entry:
            return None

        timestamp, scores = entry

        if (time.time() - timestamp) > self._ttl:

            del self._store[key]
            return None

        return scores

    def set(
        self,
        *,
        reranker: str,
        query: str,
        texts: List[str],
        scores: List[float],
    ) -> None:

        key = self._build_key(reranker, query, texts)

        self._store[key] = (time.time(), scores)