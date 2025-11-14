# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import List, Dict, Any, Optional

from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.web_document import WebDocument
from intergrax.websearch.providers.base import WebSearchProvider
from intergrax.websearch.providers.google_cse_provider import GoogleCSEProvider
from intergrax.websearch.providers.bing_provider import BingWebProvider
from intergrax.websearch.pipeline.search_and_read import SearchAndReadPipeline


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
        default_locale: str = "pl-PL",
        default_region: str = "pl-PL",
        default_language: str = "pl",
        default_safe_search: bool = True,
        max_text_chars: int = 4000,
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
        """
        self.default_top_k = default_top_k
        self.default_locale = default_locale
        self.default_region = default_region
        self.default_language = default_language
        self.default_safe_search = default_safe_search
        self.max_text_chars = max_text_chars

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

    def _serialize_web_document(self, doc: WebDocument) -> Dict[str, Any]:
        """
        Converts a WebDocument into a dict suitable for LLM prompts and logging.
        """
        page = doc.page
        hit = doc.hit

        full_text = page.text or ""
        if self.max_text_chars and len(full_text) > self.max_text_chars:
            text = full_text[: self.max_text_chars]
        else:
            text = full_text

        return {
            "provider": hit.provider,
            "rank": hit.rank,
            "source_rank": doc.source_rank,
            "quality_score": doc.quality_score,
            "title": page.title or hit.title,
            "url": hit.url,
            "snippet": hit.snippet,
            "description": page.description,
            "lang": page.lang,
            "domain": hit.domain(),
            "published_at": hit.published_at.isoformat() if hit.published_at else None,
            "fetched_at": page.fetched_at.isoformat(),
            "text": text,
        }

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
        serialize: bool = True,
    ) -> List[Any]:
        """
        Executes the full web search pipeline asynchronously.

        Returns:
          - list of serialized dicts if serialize=True,
          - list of WebDocument objects if serialize=False.
        """
        spec = self.build_query_spec(
            query=query,
            top_k=top_k,
            locale=locale,
            region=region,
            language=language,
            safe_search=safe_search,
        )

        docs = await self.pipeline.run(
            spec,
            top_n_fetch=top_n_fetch if top_n_fetch is not None else spec.top_k,
        )

        if serialize:
            return [self._serialize_web_document(d) for d in docs]
        return docs

    def search_sync(
        self,
        query: str,
        top_k: Optional[int] = None,
        locale: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        safe_search: Optional[bool] = None,
        top_n_fetch: Optional[int] = None,
        serialize: bool = True,
    ) -> List[Any]:
        """
        Synchronous wrapper for environments without an existing event loop.

        NOTE:
          Do not use this method inside Jupyter or any environment with
          a running event loop. In such cases, use 'search_async' instead.
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
                serialize=serialize,
            )
        )
