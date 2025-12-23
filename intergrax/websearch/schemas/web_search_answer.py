# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from intergrax.llm.messages import ChatMessage
from intergrax.websearch.schemas.web_search_result import WebSearchResult


@dataclass(frozen=True)
class WebSearchAnswer:
    """
    Typed result of WebSearchAnswerer.

    Fields:
      - answer: final model answer
      - messages: LLM-ready messages used to generate answer
      - web_results: typed web search results used as sources/context
    """
    answer: str
    messages: List[ChatMessage]
    web_results: List[WebSearchResult]
