# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.websearch.cache.query_cache import InMemoryQueryCache, QueryCacheKey
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.web_document import WebDocument
from intergrax.websearch.providers.base import WebSearchProvider
from intergrax.websearch.providers.google_cse_provider import GoogleCSEProvider
from intergrax.websearch.providers.bing_provider import BingWebProvider
from intergrax.websearch.pipeline.search_and_read import SearchAndReadPipeline
from intergrax.websearch.schemas.web_search_result import WebSearchResult


class WebSearchExecutor:
    """
    High-level, configurable web search executor.

    Responsibilities:
      - construct QuerySpec from a raw query and configuration,
      - execute SearchAndReadPipeline with chosen providers,
      - convert WebDocument objects into LLM-friendly dicts.

    This class is the main entry point for:
      - notebooks,
      - LangGraph nodes,
      - any other orchestration code that needs web search results.
    """

    def __init__(
        self,
        providers: Optional[List[WebSearchProvider]] = None,
        enable_google_cse: bool = True,
        enable_bing_web: bool = True,
        http_rate_per_sec: float = 2.0,
        http_capacity: int = 5,
        default_top_k: int = 8,
        default_locale=GLOBAL_SETTINGS.default_locale,
        default_region=GLOBAL_SETTINGS.default_region,
        default_language=GLOBAL_SETTINGS.default_language,
        default_safe_search: bool = True,
        max_text_chars: int = 4000,
        query_cache: Optional[InMemoryQueryCache] = None,
    ) -> None:
        """
        Parameters:
          providers        : explicit list of providers. If None, they are built from flags.
          enable_google_cse: if True and no explicit providers, include Google CSE.
          enable_bing_web  : if True and no explicit providers, include Bing Web Search.
          http_rate_per_sec: HTTP token refill rate for the pipeline.
          http_capacity    : HTTP token bucket capacity.
          default_top_k    : default number of search hits to request.
          default_locale   : default locale for QuerySpec.
          default_region   : default region for QuerySpec.
          default_language : default language for QuerySpec.
          default_safe_search: default safe search flag.
          max_text_chars   : max length of extracted text per document in serialized output.
          query_cache      : optional in-memory query-level cache for serialized results.
        """
        self.default_top_k = default_top_k
        self.default_locale = default_locale
        self.default_region = default_region
        self.default_language = default_language
        self.default_safe_search = default_safe_search
        self.max_text_chars = max_text_chars
        self._query_cache = query_cache

        if providers is None:
            providers = self._build_default_providers(
                enable_google_cse=enable_google_cse,
                enable_bing_web=enable_bing_web,
            )

        if not providers:
            raise ValueError(
                "WebSearchExecutor: no providers configured. "
                "Check API key configuration or pass providers explicitly."
            )

        self.pipeline = SearchAndReadPipeline(
            providers=providers,
            http_rate_per_sec=http_rate_per_sec,
            http_capacity=http_capacity,
        )
        self._provider_signature = self._build_provider_signature(providers)

    @staticmethod
    def _build_default_providers(
        enable_google_cse: bool,
        enable_bing_web: bool,
    ) -> List[WebSearchProvider]:
        providers: List[WebSearchProvider] = []

        if enable_google_cse:
            try:
                providers.append(GoogleCSEProvider())
            except Exception:
                # Missing configuration or init error; skip.
                pass

        if enable_bing_web:
            try:
                providers.append(BingWebProvider())
            except Exception:
                # Missing configuration or init error; skip.
                pass

        return providers

    @staticmethod
    def _build_provider_signature(providers: List[WebSearchProvider]) -> str:
        """
        Builds a simple, deterministic signature of the provider configuration.

        Used as part of the cache key so that changing providers (e.g. enabling/disabling
        Bing or Google CSE) naturally invalidates previous cached entries.
        """
        names = sorted(p.__class__.__name__ for p in providers)
        return "+".join(names)


    def _serialize_web_document(self, doc: WebDocument) -> WebSearchResult:
        """
        Converts a WebDocument into a list suitable for LLM prompts and logging.
        """
        page = doc.page
        hit = doc.hit

        full_text = page.text or ""
        if self.max_text_chars and len(full_text) > self.max_text_chars:
            text = full_text[: self.max_text_chars]
        else:
            text = full_text

        return WebSearchResult(
            provider=hit.provider,
            rank=hit.rank,
            source_rank=doc.source_rank,
            quality_score=doc.quality_score,
            title=(page.title or hit.title or ""),
            url=hit.url,
            snippet=hit.snippet,
            description=page.description,
            lang=page.lang,
            domain=hit.domain(),
            published_at=hit.published_at.isoformat() if hit.published_at else None,
            fetched_at=page.fetched_at.isoformat(),
            text=text,
            document=doc,
        )


    def build_query_spec(
        self,
        query: str,
        top_k: Optional[int] = None,
        locale: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        safe_search: Optional[bool] = None,
    ) -> QuerySpec:
        """
        Builds a QuerySpec using executor defaults and optional overrides.
        """
        return QuerySpec(
            query=query,
            top_k=top_k if top_k is not None else self.default_top_k,
            locale=locale or self.default_locale,
            region=region or self.default_region,
            language=language or self.default_language,
            safe_search=self.default_safe_search if safe_search is None else safe_search,
        )

    async def search_async(
        self,
        query: str,
        top_k: Optional[int] = None,
        locale: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        safe_search: Optional[bool] = None,
        top_n_fetch: Optional[int] = None,
    ) -> List[WebSearchResult]:
        """
        Executes the full web search pipeline asynchronously.

        Returns:
          - list of WebDocument objects

        When a query cache is configured and active:
          - attempts to return cached serialized results when available and valid.
        """
        spec = self.build_query_spec(
            query=query,
            top_k=top_k,
            locale=locale,
            region=region,
            language=language,
            safe_search=safe_search,
        )

        final_top_n = top_n_fetch if top_n_fetch is not None else spec.top_k

        cache_key: Optional[QueryCacheKey] = None
        if self._query_cache is not None:
            cache_key = QueryCacheKey(
                query=spec.query,
                top_k=final_top_n,
                locale=spec.locale or "",
                region=spec.region or "",
                language=spec.language or "",
                safe_search=spec.safe_search,
                provider_signature=self._provider_signature,
            )
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                return cached

        docs = await self.pipeline.run(
            spec,
            top_n_fetch=final_top_n,
        )

        serialized_docs = [self._serialize_web_document(d) for d in docs]

        if cache_key is not None and self._query_cache is not None:
            self._query_cache.set(cache_key, serialized_docs)

        return serialized_docs

    def search_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        locale: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        safe_search: Optional[bool] = None,
        top_n_fetch: Optional[int] = None,
    ) -> List[WebSearchResult]:
        """
        Synchronous wrapper for environments without an existing event loop.
        """
        import asyncio

        return asyncio.run(
            self.search_async(
                query=query,
                top_k=top_k,
                locale=locale,
                region=region,
                language=language,
                safe_search=safe_search,
                top_n_fetch=top_n_fetch,
            )
        )
