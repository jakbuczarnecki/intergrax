# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class PageContent:
    """
    Represents the fetched and optionally extracted content of a web page.

    This class encapsulates both raw HTML and derived metadata, 
    allowing post-processing stages (extraction, readability, deduplication)
    to work independently of the original HTTP layer.

    Fields:
      final_url       : resolved final URL after redirects
      status_code     : HTTP response code
      html            : full HTML source of the page (if available)
      text            : extracted plain text content (after cleaning)
      title           : extracted <title> tag or OG title
      description     : meta or OG description
      lang            : detected or declared language (ISO 639-1)
      og              : extracted Open Graph tags
      schema_org      : parsed JSON-LD or microdata (subset)
      fetched_at      : UTC timestamp of when the page was fetched
      robots_allowed  : whether fetching was allowed per robots.txt (if checked)
      content_bytes   : size of the HTTP body in bytes
      is_paywalled    : heuristic flag if the page is detected as paywalled
      extra           : reserved for future per-provider metadata (headers, mime, etc.)
    """
    final_url: str
    status_code: int
    html: Optional[str]
    text: Optional[str]
    title: Optional[str]
    description: Optional[str]
    lang: Optional[str]
    og: Dict[str, Any] = field(default_factory=dict)
    schema_org: Dict[str, Any] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    robots_allowed: Optional[bool] = None
    content_bytes: Optional[int] = None
    is_paywalled: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def has_content(self) -> bool:
        """
        Returns True if the page contains non-empty text or HTML.
        Used to filter out failed or empty fetches.
        """
        return bool((self.text and self.text.strip()) or (self.html and self.html.strip()))

    def short_summary(self, length: int = 200) -> str:
        """
        Returns a truncated text snippet useful for logging and debugging.
        """
        if not self.text:
            return ""
        return (self.text[:length] + "...") if len(self.text) > length else self.text

    def content_length_kb(self) -> float:
        """
        Returns the approximate size of the content in kilobytes.
        """
        if self.content_bytes is None:
            if self.html:
                return len(self.html.encode("utf-8")) / 1024
            return 0.0
        return self.content_bytes / 1024.0
