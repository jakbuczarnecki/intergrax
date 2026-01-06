# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Any, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig
from intergrax.websearch.schemas.web_search_result import WebSearchResult
from intergrax.websearch.service.websearch_config import WebSearchConfig
from intergrax.websearch.service.websearch_context_generator import create_websearch_context_generator




@dataclass
class WebSearchPromptBundle:
    """
    Container for prompt elements related to web search:

    - context_messages: system-level messages injecting web search results.
    - debug_info: structured metadata for debug traces (URLs, counts, errors).
    """
    context_messages: List[ChatMessage]
    debug_info: Dict[str, Any]


class WebSearchPromptBuilder(Protocol):
    """
    Strategy interface for building the web search part of the prompt.

    You can provide a custom implementation and pass it to
    RuntimeEngine to fully control:

    - how web documents are summarized,
    - how many results are injected,
    - the exact wording of the system messages.
    """

    async def build_websearch_prompt(
        self,
        web_results: List[WebSearchResult],
        *,
        user_query: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> WebSearchPromptBundle:
        ...


class DefaultWebSearchPromptBuilder(WebSearchPromptBuilder):
    """
    Default prompt builder for web search results in Drop-In Knowledge Mode.

    Responsibilities:
    - Take a list of typed WebSearchResult returned by WebSearchExecutor.
    - Delegate to websearch module context generator (strategy-based).
    - Wrap the generated grounded context into a single system message.
    - Provide debug info: number of docs, top URLs, strategy debug.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    async def build_websearch_prompt(
        self,
        web_results: List[WebSearchResult],
        *,
        user_query: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> WebSearchPromptBundle:
        debug_info: Dict[str, Any] = {}

        if not web_results:
            return WebSearchPromptBundle(
                context_messages=[],
                debug_info=debug_info,
            )

        debug_info["num_docs"] = len(web_results)
        debug_info["top_urls"] = [d.url for d in web_results[:3] if (d.url or "").strip()]

        # Use dedicated websearch configuration if present; otherwise fallback.
        cfg: Optional[WebSearchConfig] = self._config.websearch_config
        if cfg is None:
            # Fallback to previous behavior limits, but with safe defaults.
            # Prefer an explicit WebSearchConfig default rather than re-implementing logic here.
            cfg = WebSearchConfig()

        cfg.run_id = run_id

        gen = create_websearch_context_generator(cfg)
        result = await gen.generate(web_results, user_query=user_query)

        # IMPORTANT: context injection should be a system message
        context_messages = [
            ChatMessage(
                role="system",
                content=result.context_text,
            )
        ]

        # Merge generator debug info into builder debug info
        for k, v in result.debug_info.items():
            debug_info[k] = v

        return WebSearchPromptBundle(
            context_messages=context_messages,
            debug_info=debug_info,
        )
