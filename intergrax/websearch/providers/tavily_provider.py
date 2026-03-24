# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

from tavily.errors import (
    BadRequestError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit
from intergrax.websearch.providers.base import WebSearchProvider


class TavilyProvider(WebSearchProvider):
    """
    Tavily web search provider.

    Environment variables:
      TAVILY_API_KEY : API key (required)

    Notes:
      - Tavily supports language hints via the query itself.
      - Freshness filtering is available via 'time_range' parameter (day, week, month, year).
      - include_answer controls whether an AI-generated answer is included.
    """

    name: str = "tavily"
    _PAGE_CAP: int = 20
    _DEFAULT_SEARCH_DEPTH: str = "basic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: Optional[str] = None,
    ) -> None:
        from tavily import TavilyClient

        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.search_depth = search_depth or self._DEFAULT_SEARCH_DEPTH

        if not self.api_key:
            raise ValueError("TavilyProvider: missing API key (TAVILY_API_KEY).")

        self._client = TavilyClient(api_key=self.api_key)

    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_language": False,
            "supports_freshness": True,
            "max_page_size": self._PAGE_CAP,
        }

    def _map_freshness(self, freshness: Optional[str]) -> Optional[str]:
        """Map QuerySpec freshness values to Tavily time_range parameter."""
        if not freshness:
            return None
        mapping = {
            "day": "day",
            "week": "week",
            "month": "month",
            "year": "year",
        }
        return mapping.get(freshness.lower())

    def _to_hit(self, item: Dict[str, Any], spec: QuerySpec, rank: int) -> Optional[SearchHit]:
        url = item.get("url") or ""
        parsed = urlparse(url)
        if not (parsed.scheme and parsed.netloc):
            return None

        title = item.get("title", "") or ""
        snippet = item.get("content", "") or None

        published_at: Optional[datetime] = None
        date_str = item.get("published_date")
        if date_str:
            try:
                published_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                published_at = None

        score = item.get("score")

        return SearchHit(
            provider=self.name,
            query_issued=spec.query,
            rank=rank,
            title=title,
            url=url,
            snippet=snippet,
            published_at=published_at,
            source_type="web",
            extra={"score": score} if score is not None else {},
        )

    def search(self, spec: QuerySpec) -> List[SearchHit]:
        kwargs: Dict[str, Any] = {
            "query": spec.normalized_query(),
            "max_results": spec.capped_top_k(self._PAGE_CAP),
            "search_depth": self.search_depth,
        }

        time_range = self._map_freshness(spec.freshness)
        if time_range:
            kwargs["time_range"] = time_range

        try:
            response = self._client.search(**kwargs)
        except (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
            TavilyTimeoutError,
            requests.RequestException,
        ):
            return []

        results = response.get("results", []) or []
        hits: List[SearchHit] = []
        for i, item in enumerate(results, start=1):
            hit = self._to_hit(item, spec, i)
            if hit:
                hits.append(hit)
        return hits

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
