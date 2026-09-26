# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class SearchHit:
    """Provider-agnostic metadata for a single search result entry."""

    provider: str
    query_issued: str
    rank: int
    title: str
    url: str
    snippet: str | None = None
    displayed_link: str | None = None
    published_at: datetime | None = None
    source_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        parsed = urlparse(self.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"invalid search hit url: {self.url}")


__all__ = ["SearchHit"]
