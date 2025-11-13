# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class QuerySpec:
    """
    Canonical search query specification used by all web search providers.
    Keep this model minimal, provider-agnostic, and stable.

    Fields:
      query       : raw user query (required)
      top_k       : desired number of top results per provider (provider may cap this)
      locale      : BCP 47 locale (e.g., "pl-PL") used for UI language hints
      region      : market/region code (e.g., "PL", "US" or "pl-PL" for certain APIs)
      language    : ISO 639-1 content language code (e.g., "pl", "en"), 
                    used to restrict results by language when the provider supports it
      freshness   : optional recency constraint (provider-specific semantics)
      site_filter : optional site restriction (e.g., "site:example.com")
      safe_search : request safe-search filtering where available
    """
    query: str
    top_k: int = 8
    locale: Optional[str] = None
    region: Optional[str] = None
    language: Optional[str] = None
    freshness: Optional[str] = None
    site_filter: Optional[str] = None
    safe_search: bool = True

    def normalized_query(self) -> str:
        """
        Returns the query string with an applied site filter (if present).
        Providers that accept a combined 'q' parameter can use this directly.
        """
        q = (self.query or "").strip()
        if self.site_filter:
            q = f"{q} {self.site_filter}".strip()
        return q

    def capped_top_k(self, provider_cap: int) -> int:
        """
        Returns a provider-safe 'top_k' that never exceeds the provider's cap and is >= 1.
        """
        k = max(1, int(self.top_k or 1))
        return min(k, max(1, int(provider_cap or k)))
