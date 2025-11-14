# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import time


@dataclass(frozen=True)
class QueryCacheKey:
    """
    Immutable key describing a unique web search configuration.

    Two searches with the same key should be considered equivalent
    and safe to reuse cached results.
    """
    query: str
    top_k: int
    locale: str
    region: str
    language: str
    safe_search: bool
    provider_signature: str  # identifies enabled providers configuration

    def as_tuple(self) -> Tuple[Any, ...]:
        return (
            self.query,
            self.top_k,
            self.locale,
            self.region,
            self.language,
            self.safe_search,
            self.provider_signature,
        )


@dataclass
class QueryCacheEntry:
    """
    Stored cache value for a given query key.
    """
    documents: List[Dict[str, Any]]
    created_at: float


class InMemoryQueryCache:
    """
    Simple in-memory query cache with optional TTL and max size.

    This cache stores already serialized web documents:
      List[Dict[str, Any]]

    It is intentionally simple and single-process only:
    - suitable for notebooks and local development
    - can be replaced with a distributed backend later (Redis, etc.)
    """

    def __init__(
        self,
        max_entries: int = 256,
        ttl_seconds: Optional[int] = 600,
    ) -> None:
        """
        Parameters:
          max_entries: maximum number of entries stored in memory
          ttl_seconds: time-to-live in seconds; if None → never expire
        """
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._store: Dict[Tuple[Any, ...], QueryCacheEntry] = {}

    def _is_expired(self, entry: QueryCacheEntry) -> bool:
        if self._ttl_seconds is None:
            return False
        return (time.time() - entry.created_at) > self._ttl_seconds

    def get(self, key: QueryCacheKey) -> Optional[List[Dict[str, Any]]]:
        k = key.as_tuple()
        entry = self._store.get(k)
        if entry is None:
            return None
        if self._is_expired(entry):
            # Remove expired entry
            self._store.pop(k, None)
            return None
        return entry.documents

    def set(self, key: QueryCacheKey, documents: List[Dict[str, Any]]) -> None:
        if len(self._store) >= self._max_entries:
            # Very simple eviction: remove oldest entry
            oldest_key = None
            oldest_ts = float("inf")
            for kk, entry in self._store.items():
                if entry.created_at < oldest_ts:
                    oldest_ts = entry.created_at
                    oldest_key = kk
            if oldest_key is not None:
                self._store.pop(oldest_key, None)

        self._store[key.as_tuple()] = QueryCacheEntry(
            documents=documents,
            created_at=time.time(),
        )
