# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
import time

from intergrax.websearch.schemas.web_search_result import WebSearchResult


def _norm_str(value: Optional[str], default: str) -> str:
    v = (value or "").strip()
    return v if v else default


@dataclass(frozen=True)
class QueryCacheKey:
    """
    Immutable key describing a unique web search configuration.

    Two searches with the same key should be considered equivalent
    and safe to reuse cached results.

    NOTE:
    This key MUST use canonicalized values (no None) to avoid cache misses.
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

    @staticmethod
    def from_params(
        *,
        query: str,
        top_k: Optional[int],
        locale: Optional[str],
        region: Optional[str],
        language: Optional[str],
        safe_search: Optional[bool],
        provider_signature: str,
        default_top_k: int = 8,
        default_locale: str = "en_US",
        default_region: str = "US",
        default_language: str = "en",
        default_safe_search: bool = True,
    ) -> "QueryCacheKey":
        """
        Build a canonical cache key from optional parameters.
        """
        q = (query or "").strip()
        return QueryCacheKey(
            query=q,
            top_k=int(top_k if top_k is not None else default_top_k),
            locale=_norm_str(locale, default_locale),
            region=_norm_str(region, default_region),
            language=_norm_str(language, default_language),
            safe_search=bool(safe_search if safe_search is not None else default_safe_search),
            provider_signature=(provider_signature or "").strip(),
        )


@dataclass
class QueryCacheEntry:
    """
    Stored cache value for a given query key.
    """
    documents: List[WebSearchResult]
    created_at: float


class InMemoryQueryCache:
    """
    Simple in-memory query cache with optional TTL and max size.

    This cache stores typed web search results:
      List[WebSearchResult]

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
        self._max_entries = int(max_entries)
        self._ttl_seconds = ttl_seconds
        self._store: Dict[Tuple[Any, ...], QueryCacheEntry] = {}

    @property
    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def _is_expired(self, entry: QueryCacheEntry) -> bool:
        if self._ttl_seconds is None:
            return False
        return (time.time() - entry.created_at) > self._ttl_seconds

    def get(self, key: QueryCacheKey) -> Optional[List[WebSearchResult]]:
        k = key.as_tuple()
        entry = self._store.get(k)
        if entry is None:
            return None
        if self._is_expired(entry):
            self._store.pop(k, None)
            return None
        return entry.documents

    def set(self, key: QueryCacheKey, documents: List[WebSearchResult]) -> None:
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
