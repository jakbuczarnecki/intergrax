# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Mapping, Any
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit

class WebSearchProvider(ABC):
    """
    Base interface for all web search providers (Google, Bing, DuckDuckGo, Reddit, News, etc.).

    Responsibilities:
      - Accept a provider-agnostic QuerySpec.
      - Return a ranked list of SearchHit items.
      - Expose minimal capabilities for feature negotiation (language, freshness).

    Design notes:
      - Keep this interface stable; providers should adapt to it, not the other way around.
      - Avoid leaking provider-specific parameters here; put them in provider constructors.
    """

    name: str = "base"

    @abstractmethod
    def search(self, spec: QuerySpec) -> List[SearchHit]:
        """
        Executes a single search request.
        Must return a list of SearchHit with 1-based 'rank' ordering.
        Implementations should:
          - honor QuerySpec.top_k with provider-side caps,
          - include 'provider' and 'query_issued' fields in hits,
          - sanitize/validate URLs.
        """
        raise NotImplementedError

    def capabilities(self) -> Mapping[str, Any]:
        """
        Returns a static capability map for feature negotiation.
        Keys are stable, values are simple scalars/flags.

        Example keys:
          supports_language: bool
          supports_freshness: bool
          max_page_size: int
        """
        return {
            "supports_language": False,
            "supports_freshness": False,
            "max_page_size": 10,
        }

    def close(self) -> None:
        """
        Optional resource cleanup (HTTP sessions, clients, caches).
        Providers that own such resources should override this method.
        """
        return None
