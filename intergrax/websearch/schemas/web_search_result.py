# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.websearch.schemas.web_document import WebDocument


@dataclass(frozen=True)
class WebSearchResult:
    provider: str
    rank: int
    source_rank: Optional[int]
    quality_score: Optional[float]

    title: str
    url: str
    snippet: Optional[str]
    description: Optional[str]

    lang: Optional[str]
    domain: Optional[str]

    published_at: Optional[str]
    fetched_at: str

    text: str

    document: WebDocument