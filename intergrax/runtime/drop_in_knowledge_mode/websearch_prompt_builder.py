# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Mapping, Any

from intergrax.llm.conversational_memory import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.config import RuntimeConfig


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
    DropInKnowledgeRuntime to fully control:

    - how web documents are summarized,
    - how many results are injected,
    - the exact wording of the system messages.
    """

    def build_websearch_prompt(
        self,
        web_docs: List[Dict[str, Any]],
    ) -> WebSearchPromptBundle:
        ...
    

class DefaultWebSearchPromptBuilder(WebSearchPromptBuilder):
    """
    Default prompt builder for web search results in Drop-In Knowledge Mode.

    Responsibilities:
    - Take a list of web documents (dict-like objects) returned by WebSearchExecutor.
    - Build a single system-level message that lists titles, URLs and snippets.
    - Provide basic debug info: number of docs and top URLs.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def build_websearch_prompt(
        self,
        web_docs: List[Dict[str, Any]],
    ) -> WebSearchPromptBundle:
        debug_info: Dict[str, Any] = {}

        if not web_docs:
            return WebSearchPromptBundle(
                context_messages=[],
                debug_info=debug_info,
            )

        debug_info["num_docs"] = len(web_docs)
        debug_info["top_urls"] = [
            d.get("url") for d in web_docs[:3] if isinstance(d, dict)
        ]

        lines: List[str] = [
            "The following web search results may be relevant. "
            "Use them together with uploaded documents and chat history. "
            "Treat them as external context, and never fabricate additional "
            "facts that are not supported by these sources or your base knowledge."
        ]

        max_docs = self._config.max_docs_per_query

        for idx, doc in enumerate(web_docs[:max_docs], start=1):
            if not isinstance(doc, dict):
                continue

            title = doc.get("title") or "(no title)"
            url = doc.get("url") or ""
            snippet = doc.get("snippet") or doc.get("text") or ""

            lines.append(f"\n[{idx}] {title}\nURL: {url}\nSnippet: {snippet}")

        context_text = "\n".join(lines)

        context_messages = [
            ChatMessage(
                role="system",
                content=context_text,
            )
        ]

        return WebSearchPromptBundle(
            context_messages=context_messages,
            debug_info=debug_info,
        )