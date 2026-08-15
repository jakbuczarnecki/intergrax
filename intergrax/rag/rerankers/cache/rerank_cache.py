# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

from intergrax.rag.rerankers.cache.base_rerank_cache import (
    BaseRerankCache,
    RerankIdentityKey,
)


class RerankCache(BaseRerankCache):
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

    def _normalize_text(self, text: str) -> str:
        """
        Normalize candidate text before hashing to avoid
        cache misses caused by whitespace differences.
        """
        return text.strip()

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def _hash_documents(
        self,
        texts: List[str],
        identity_keys: Sequence[RerankIdentityKey] | None = None,
    ) -> str:
        """
        Deterministically hash candidate texts using incremental hashing.
        Avoids delimiter ambiguity and large intermediate strings.
        """

        hasher = hashlib.sha256()
        if identity_keys is not None and len(identity_keys) != len(texts):
            raise ValueError("identity_keys and texts length mismatch")

        for index, text in enumerate(texts):
            if identity_keys is not None:
                identity = json.dumps(
                    identity_keys[index],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                hasher.update(len(identity).to_bytes(8, "big"))
                hasher.update(identity)

            normalized = self._normalize_text(text)

            encoded = normalized.encode("utf-8")

            # length prefix prevents boundary ambiguity
            length_prefix = len(encoded).to_bytes(8, "big")

            hasher.update(length_prefix)
            hasher.update(encoded)

        return hasher.hexdigest()

    def _build_key(
        self,
        reranker: str,
        query: str,
        texts: List[str],
        identity_keys: Sequence[RerankIdentityKey] | None = None,
    ) -> str:

        q_hash = self._hash_query(query)
        d_hash = self._hash_documents(texts, identity_keys)

        return f"rerank:{reranker}:{q_hash}:{d_hash}"

    def get(
        self,
        *,
        reranker: str,
        query: str,
        texts: List[str],
        identity_keys: Sequence[RerankIdentityKey] | None = None,
    ) -> Optional[List[float]]:

        key = self._build_key(reranker, query, texts, identity_keys)

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
        identity_keys: Sequence[RerankIdentityKey] | None = None,
        scores: List[float],
    ) -> None:

        key = self._build_key(reranker, query, texts, identity_keys)

        self._store[key] = (time.time(), scores)