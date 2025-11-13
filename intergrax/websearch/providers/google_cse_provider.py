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


class GoogleCSEProvider(WebSearchProvider):
    """
    Google Custom Search (CSE) provider (REST API).

    Environment variables:
      GOOGLE_CSE_API_KEY : API key
      GOOGLE_CSE_CX      : Search engine ID

    Notes:
      - Google CSE caps 'num' to 10 per request.
      - Language filtering uses 'lr' (e.g., 'lang_pl') and/or 'hl' (UI lang).
      - Freshness is not natively supported; ignore spec.freshness here.
    """

    name: str = "google_cse"
    _ENDPOINT: str = "https://www.googleapis.com/customsearch/v1"
    _PAGE_CAP: int = 10
    _DEFAULT_TIMEOUT: int = 20

    def __init__(
        self,
        api_key: Optional[str] = None,
        cx: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_CSE_API_KEY", "")
        self.cx = cx or os.getenv("GOOGLE_CSE_CX", "")
        self.session = session or requests.Session()
        self.timeout = int(timeout or self._DEFAULT_TIMEOUT)

        if not self.api_key:
            raise ValueError("GoogleCSEProvider: missing API key (GOOGLE_CSE_API_KEY).")
        if not self.cx:
            raise ValueError("GoogleCSEProvider: missing CX (GOOGLE_CSE_CX).")

    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_language": True,    # via 'lr' and 'hl'
            "supports_freshness": False,  # not exposed in CSE
            "max_page_size": self._PAGE_CAP,
        }

    def _build_params(self, spec: QuerySpec) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "key": self.api_key,
            "cx": self.cx,
            "q": spec.normalized_query(),
            "num": spec.capped_top_k(self._PAGE_CAP),
            "safe": "active" if spec.safe_search else "off",
        }

        # UI language (hl): affects interface and possibly ranking hints.
        if spec.locale:
            params["hl"] = spec.locale

        # Content language (lr): restrict results to a specific language (if provided).
        # Google expects 'lr=lang_<code>', e.g. 'lang_pl', 'lang_en'.
        if spec.language:
            params["lr"] = f"lang_{spec.language.lower()}"

        # Region is not directly supported here; could be handled via custom engines or site filters.
        # Freshness ignored (no native param in CSE).

        return params

    def _to_hit(self, item: Dict[str, Any], spec: QuerySpec, rank: int) -> Optional[SearchHit]:
        # Validate URL early; drop entries without a valid absolute URL.
        url = item.get("link", "") or ""
        parsed = urlparse(url)
        if not (parsed.scheme and parsed.netloc):
            return None

        title = item.get("title", "") or ""
        snippet = item.get("snippet", "") or None
        display_link = item.get("displayLink", "") or None

        # Published time is rarely present in CSE items; attempt best-effort extraction from pagemap.
        published_at: Optional[datetime] = None
        pagemap = item.get("pagemap") or {}
        metatags = (pagemap.get("metatags") or [])
        # Try common keys
        for meta in metatags:
            dt = meta.get("article:published_time") or meta.get("og:updated_time") or meta.get("og:published_time")
            if dt:
                try:
                    published_at = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                    break
                except Exception:
                    published_at = None

        # Infer coarse type from MIME if provided
        source_type = None
        mime = item.get("mime") or ""
        if mime:
            if "pdf" in mime.lower():
                source_type = "pdf"
            else:
                source_type = "web"

        return SearchHit(
            provider=self.name,
            query_issued=spec.query,
            rank=rank,
            title=title,
            url=url,
            snippet=snippet,
            displayed_link=display_link,
            published_at=published_at,
            source_type=source_type,
            extra={"mime": mime} if mime else {},
        )

    def search(self, spec: QuerySpec) -> List[SearchHit]:
        params = self._build_params(spec)
        try:
            r = self.session.get(self._ENDPOINT, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            # Fail fast with empty results; upstream pipeline can log/handle.
            return []

        items = data.get("items") or []
        hits: List[SearchHit] = []
        for i, it in enumerate(items, start=1):
            hit = self._to_hit(it, spec, i)
            if hit:
                hits.append(hit)

        # Ensure stable 1-based rank ordering (some items may be filtered out).
        for idx, h in enumerate(hits, start=1):
            # dataclass is frozen=False by default here; ensure safe mutation
            if h.rank != idx:
                hits[idx - 1] = SearchHit(
                    provider=h.provider,
                    query_issued=h.query_issued,
                    rank=idx,
                    title=h.title,
                    url=h.url,
                    snippet=h.snippet,
                    displayed_link=h.displayed_link,
                    published_at=h.published_at,
                    source_type=h.source_type,
                    extra=h.extra,
                )
        return hits

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
