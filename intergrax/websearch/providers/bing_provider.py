# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import os
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit
from intergrax.websearch.providers.base import WebSearchProvider


class BingWebProvider(WebSearchProvider):
    """
    Bing Web Search (v7) provider (REST API).

    Environment variables:
      BING_SEARCH_V7_API_KEY : API key

    Notes:
      - Bing supports language (setLang) and region (mkt) filtering.
      - Freshness filtering is available via 'freshness' parameter (Day, Week, Month).
      - SafeSearch has values: Off, Moderate, Strict.
    """

    name: str = "bing_web"
    _ENDPOINT: str = "https://api.bing.microsoft.com/v7.0/search"
    _DEFAULT_TIMEOUT: int = 20
    _PAGE_CAP: int = 50

    def __init__(
        self,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("BING_SEARCH_V7_API_KEY", "")
        self.session = session or requests.Session()
        self.timeout = int(timeout or self._DEFAULT_TIMEOUT)

        if not self.api_key:
            raise ValueError("BingWebProvider: missing API key (BING_SEARCH_V7_API_KEY).")

    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_language": True,
            "supports_freshness": True,
            "max_page_size": self._PAGE_CAP,
        }

    def _build_headers(self) -> Dict[str, str]:
        return {"Ocp-Apim-Subscription-Key": self.api_key}

    def _build_params(self, spec: QuerySpec) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "q": spec.normalized_query(),
            "count": spec.capped_top_k(self._PAGE_CAP),
            "safeSearch": "Strict" if spec.safe_search else "Off",
        }

        if spec.region:
            params["mkt"] = spec.region

        if spec.language:
            params["setLang"] = spec.language.lower()

        if spec.freshness:
            # Accepted values: "Day", "Week", "Month"
            val = spec.freshness.capitalize()
            if val in ("Day", "Week", "Month"):
                params["freshness"] = val

        return params

    def _to_hit(self, item: Dict[str, Any], spec: QuerySpec, rank: int) -> Optional[SearchHit]:
        url = item.get("url") or ""
        parsed = urlparse(url)
        if not (parsed.scheme and parsed.netloc):
            return None

        title = item.get("name", "") or ""
        snippet = item.get("snippet", "") or None
        display_link = item.get("displayUrl", "") or None

        published_at: Optional[datetime] = None
        date_str = item.get("dateLastCrawled")
        if date_str:
            try:
                published_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                published_at = None

        language = item.get("language") or None

        return SearchHit(
            provider=self.name,
            query_issued=spec.query,
            rank=rank,
            title=title,
            url=url,
            snippet=snippet,
            displayed_link=display_link,
            published_at=published_at,
            source_type="web",
            extra={"language": language} if language else {},
        )

    def search(self, spec: QuerySpec) -> List[SearchHit]:
        headers = self._build_headers()
        params = self._build_params(spec)

        try:
            r = self.session.get(self._ENDPOINT, headers=headers, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException:
            return []

        web_pages = (data.get("webPages") or {}).get("value", []) or []
        hits: List[SearchHit] = []
        for i, it in enumerate(web_pages, start=1):
            hit = self._to_hit(it, spec, i)
            if hit:
                hits.append(hit)
        return hits

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
