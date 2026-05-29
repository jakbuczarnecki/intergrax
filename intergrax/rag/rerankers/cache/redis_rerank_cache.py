# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Redis-backed cache for reranker score results.

Composition root: ``intergrax.integrations.providers.redis.create_redis_rerank_cache``.
"""

from __future__ import annotations

import hashlib
import json
from typing import List, Optional

import redis

from intergrax.rag.rerankers.cache.base_rerank_cache import BaseRerankCache


class RedisRerankCache(BaseRerankCache):
    """
    Redis-backed cache for reranker score results.
    """

    def __init__(
        self,
        *,
        redis_client: redis.Redis,
        ttl_seconds: int = 3600,
        key_prefix: str = "rerank",
    ) -> None:

        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = key_prefix

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

        return f"{self._prefix}:{reranker}:{q_hash}:{d_hash}"

    def get(
        self,
        *,
        reranker: str,
        query: str,
        texts: List[str],
    ) -> Optional[List[float]]:

        key = self._build_key(reranker, query, texts)

        value = self._redis.get(key)

        if value is None:
            return None

        return json.loads(value)

    def set(
        self,
        *,
        reranker: str,
        query: str,
        texts: List[str],
        scores: List[float],
    ) -> None:

        key = self._build_key(reranker, query, texts)

        self._redis.setex(
            key,
            self._ttl,
            json.dumps(scores),
        )