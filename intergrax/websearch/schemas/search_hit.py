# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import urlparse

@dataclass(frozen=True)
class SearchHit:
    """
    Provider-agnostic metadata for a single search result entry.

    Fields:
      provider       : provider identifier (e.g., "google_cse", "bing_web")
      query_issued   : original user query string used for this search
      rank           : 1-based rank within the provider's result list
      title          : result title as returned by the provider
      url            : canonical or direct URL to the resource
      snippet        : short textual summary shown by the provider (if any)
      displayed_link : human-readable display URL (if provided by the API)
      published_at   : optional publication datetime (if available)
      source_type    : coarse-grained type (e.g., "web", "news", "forum", "video", "pdf")
      extra          : provider-specific fields (kept for debugging/telemetry)
    """
    provider: str
    query_issued: str
    rank: int
    title: str
    url: str
    snippet: Optional[str] = None
    displayed_link: Optional[str] = None
    published_at: Optional[datetime] = None
    source_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Minimal safety checks:
        - enforce rank >= 1
        - ensure URL has a scheme and netloc
        """
        if self.rank < 1:
            object.__setattr__(self, "rank", 1)

        parsed = urlparse(self.url or "")
        if not (parsed.scheme and parsed.netloc):
            # Keep strict failure early — invalid hits should not propagate.
            raise ValueError(f"Invalid URL in SearchHit: '{self.url}'")

    def domain(self) -> str:
        """
        Returns the netloc (domain:port) part of the URL for quick grouping or scoring.
        """
        return urlparse(self.url).netloc

    def to_minimal_dict(self) -> Dict[str, Any]:
        """
        Returns a minimal, LLM-friendly representation of the hit.
        Intended for prompts and logs without leaking provider internals.
        """
        return {
            "provider": self.provider,
            "rank": self.rank,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "displayed_link": self.displayed_link,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "source_type": self.source_type,
            "domain": self.domain(),
        }
