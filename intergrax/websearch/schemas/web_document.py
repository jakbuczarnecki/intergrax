# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .search_hit import SearchHit
from .page_content import PageContent

@dataclass
class WebDocument:
    """
    Unified structure representing a fetched and processed web document.

    It connects the original search hit (provider metadata)
    with the extracted content (PageContent) and analysis results
    such as deduplication and quality scores.

    Fields:
      hit           : source SearchHit from provider API
      page          : fetched PageContent (HTML + extracted text)
      dedupe_key    : stable hash or simhash for duplicate detection
      quality_score : normalized quality metric (0–1 or arbitrary float)
      source_rank   : optional adjusted rank after multi-provider fusion
    """
    hit: SearchHit
    page: PageContent
    dedupe_key: Optional[str] = None
    quality_score: float = 0.0
    source_rank: Optional[int] = None

    def is_valid(self) -> bool:
        """
        Returns True if the document contains valid textual content and a valid URL.
        """
        return bool(self.page and self.page.has_content() and self.hit and self.hit.url)

    def merged_text(self, max_length: Optional[int] = None) -> str:
        """
        Returns combined textual content for LLM or retrieval embedding.
        """
        if not self.page or not self.page.text:
            return ""
        text = self.page.text.strip()
        if max_length and len(text) > max_length:
            return text[:max_length]
        return text

    def summary_line(self) -> str:
        """
        Returns a short one-line summary used in logs or console outputs.
        """
        t = self.page.title or self.hit.title or ""
        domain = self.hit.domain()
        score = f"{self.quality_score:.2f}"
        return f"[{domain}] {t} (score={score})"
