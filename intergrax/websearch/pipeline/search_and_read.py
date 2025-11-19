# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence

import asyncio

from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit
from intergrax.websearch.schemas.web_document import WebDocument

from intergrax.websearch.providers.base import WebSearchProvider

from intergrax.websearch.fetcher.http_fetcher import fetch_page
from intergrax.websearch.fetcher.extractor import extract_basic, extract_advanced

from intergrax.websearch.utils.rate_limit import TokenBucket
from intergrax.websearch.utils.dedupe import simple_dedupe_key


class SearchAndReadPipeline:
    """
    Orchestrates multi-provider web search, fetching, extraction,
    deduplication, and basic quality scoring.

    High-level flow:
      1) search_all(spec)         -> List[SearchHit]
      2) fetch_and_extract(hits)  -> List[WebDocument]
      3) caller passes WebDocument objects to LLM / RAG layers.

    Design goals:
      - Provider-agnostic: works with any WebSearchProvider.
      - Async fetching with rate limiting (TokenBucket).
      - Simple deduplication via text-based dedupe key.
      - Minimal and testable, with no direct LLM coupling.
    """

    def __init__(
        self,
        providers: Iterable[WebSearchProvider],
        http_rate_per_sec: float = 2.0,
        http_capacity: int = 5,
    ) -> None:
        self._providers: List[WebSearchProvider] = list(providers)
        self._bucket = TokenBucket(
            rate_per_sec=http_rate_per_sec,
            capacity=http_capacity,
        )

    @property
    def providers(self) -> Sequence[WebSearchProvider]:
        """
        Returns the current list of configured providers.
        """
        return tuple(self._providers)

    def search_all(self, spec: QuerySpec) -> List[SearchHit]:
        """
        Executes the given query against all configured providers.

        Behavior:
          - Calls each provider sequentially (providers can be rate-limited internally).
          - Merges all hits into a single list.
          - Sorts results by 'rank' within each provider, but does not
            perform complex cross-provider fusion (caller can add it later).

        Failures:
          - Provider-level exceptions are swallowed; failed providers
            contribute zero results. Logging should be added at integration time.
        """
        hits: List[SearchHit] = []
        for provider in self._providers:
            try:
                provider_hits = provider.search(spec)
                hits.extend(provider_hits)
            except Exception:
                # Intentionally silent here; integration layer should add logging if needed.
                continue

        # Preserve provider ordering and rank while still returning a deterministic list.
        hits.sort(key=lambda h: (h.provider, h.rank))
        return hits

    async def _fetch_one(self, hit: SearchHit) -> Optional[WebDocument]:
        """
        Fetches and extracts a single SearchHit into a WebDocument.

        Steps:
          - Rate-limited HTTP GET of hit.url.
          - Basic HTML extraction (title, text, metadata).
          - Quality scoring and dedupe key computation.
        """
        await self._bucket.acquire(1)

        page = await fetch_page(hit.url)
        if not page or not page.html:
            return None

        page = extract_basic(page)
        page = extract_advanced(page)

        if not page.has_content():
            return None

        doc = WebDocument(hit=hit, page=page)

        # Very simple quality heuristic:
        text_len = len(page.text or "")
        has_title = bool(page.title and page.title.strip())
        score = 0.0
        if text_len >= 400:
            score += 1.0
        if text_len >= 2000:
            score += 0.5
        if has_title:
            score += 0.5
        doc.quality_score = score

        # Deduplication key based on title + text.
        base_text = (page.title or "") + "\n" + (page.text or "")
        doc.dedupe_key = simple_dedupe_key(base_text)

        # Initial source_rank mirrors the provider-local rank.
        doc.source_rank = hit.rank

        return doc

    async def fetch_and_extract(
        self,
        hits: List[SearchHit],
        top_n_fetch: int = 8,
    ) -> List[WebDocument]:
        """
        Fetches and processes the top-N hits into WebDocument objects.

        Parameters:
          hits        : list of SearchHit returned from search_all.
          top_n_fetch : maximum number of URLs to fetch in this pass.

        Behavior:
          - Uses asyncio to fetch multiple pages concurrently.
          - Deduplicates documents based on dedupe_key.
          - Sorts results by quality_score (descending), then by source_rank (ascending).
        """
        if not hits:
            return []

        limited_hits = hits[:max(0, top_n_fetch)]
        tasks = [self._fetch_one(h) for h in limited_hits]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        docs: List[WebDocument] = [d for d in results if d and d.is_valid()]

        # Deduplicate by dedupe_key, keeping the first occurrence.
        seen_keys = set()
        unique_docs: List[WebDocument] = []
        for doc in docs:
            key = doc.dedupe_key or ""
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            unique_docs.append(doc)

        # Sort by quality_score (desc) then source_rank (asc).
        unique_docs.sort(
            key=lambda d: (-float(d.quality_score or 0.0), int(d.source_rank or 0)),
        )

        return unique_docs

    async def run(
        self,
        spec: QuerySpec,
        top_n_fetch: int = 8,
    ) -> List[WebDocument]:
        """
        Full pipeline execution for a single query.

        Steps:
          1) search_all(spec)
          2) fetch_and_extract(hits, top_n_fetch)
          3) return List[WebDocument]

        The returned documents are ready to be consumed by:
          - LLM adapters (as web context),
          - embedding pipelines (as text to index),
          - logging and inspection tools (summary_line, etc.).
        """
        hits = self.search_all(spec)
        if not hits:
            return []
        docs = await self.fetch_and_extract(hits, top_n_fetch=top_n_fetch)
        return docs

    def run_sync(
        self,
        spec: QuerySpec,
        top_n_fetch: int = 8,
    ) -> List[WebDocument]:
        """
        Synchronous convenience wrapper around the async 'run' method.

        This is intended for environments where no event loop is managed
        externally (e.g., simple scripts or sync LangGraph nodes).

        In async applications, prefer calling 'await run(...)' directly.
        """
        return asyncio.run(self.run(spec, top_n_fetch=top_n_fetch))
