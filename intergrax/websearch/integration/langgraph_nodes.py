# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any, Annotated

try:
    from langgraph.graph.message import add_messages
except ImportError:
    def add_messages(x: Any) -> Any:
        return x

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


class WebSearchState(TypedDict, total=False):
    """
    Minimal state contract for web search nodes.

    Fields:
      messages        : conversation history (LangGraph-compatible)
      user_question   : last user question (fallback for query)
      websearch_query : explicit query string for web search (optional)
      websearch_docs  : serialized web documents (ready for LLM consumption)
    """
    messages: Annotated[list, add_messages]
    user_question: Optional[str]
    websearch_query: Optional[str]
    websearch_docs: Optional[List[Dict[str, Any]]]


class WebSearchNode:
    """
    LangGraph-compatible web search node wrapper.

    This class encapsulates:
      - configuration of WebSearchExecutor (providers, defaults),
      - sync and async node methods operating on WebSearchState.

    The node does not implement search logic itself. It delegates
    to the provided WebSearchExecutor instance.
    """

    def __init__(
        self,
        executor: Optional[WebSearchExecutor] = None,
        enable_google_cse: bool = True,
        enable_bing_web: bool = True,
        default_top_k: int = 8,
        default_locale: str = GLOBAL_SETTINGS.default_locale,
        default_region: str = GLOBAL_SETTINGS.default_region,
        default_language: str = GLOBAL_SETTINGS.default_language,
        default_safe_search: bool = True,
        max_text_chars: int = 4000,
    ) -> None:
        """
        Parameters:
          executor           : externally configured WebSearchExecutor. If None, one is created.
          enable_google_cse  : used only when executor is None.
          enable_bing_web    : used only when executor is None.
          default_top_k      : used only when executor is None.
          default_locale     : used only when executor is None.
          default_region     : used only when executor is None.
          default_language   : used only when executor is None.
          default_safe_search: used only when executor is None.
          max_text_chars     : used only when executor is None.
        """
        if executor is not None:
            self.executor = executor
        else:
            self.executor = WebSearchExecutor(
                enable_google_cse=enable_google_cse,
                enable_bing_web=enable_bing_web,
                default_top_k=default_top_k,
                default_locale=default_locale,
                default_region=default_region,
                default_language=default_language,
                default_safe_search=default_safe_search,
                max_text_chars=max_text_chars,
            )

    def _extract_query(self, state: WebSearchState) -> str:
        """
        Extracts the search query from the node state.
        Preference order:
          1) websearch_query
          2) user_question
        """
        return (state.get("websearch_query") or state.get("user_question") or "").strip()

    def run(self, state: WebSearchState) -> WebSearchState:
        """
        Synchronous node method - suitable for non-async environments.

        In environments with a running event loop (e.g. Jupyter),
        prefer using 'run_async' directly.
        """
        query = self._extract_query(state)
        if not query:
            state["websearch_docs"] = []
            return state

        docs = self.executor.search_sync(
            query=query,
            top_k=None,       # use executor defaults
            top_n_fetch=None, # use top_k
            serialize=True,
        )

        state["websearch_docs"] = docs
        return state

    async def run_async(self, state: WebSearchState) -> WebSearchState:
        """
        Async node method - safe to use in environments with an existing
        event loop (Jupyter, async web frameworks, LangGraph runtimes).
        """
        query = self._extract_query(state)
        if not query:
            state["websearch_docs"] = []
            return state

        docs = await self.executor.search_async(
            query=query,
            top_k=None,       # use executor defaults
            top_n_fetch=None, # use top_k
            serialize=True,
        )

        state["websearch_docs"] = docs
        return state


# Default, module-level node instance for convenience and backward compatibility
_DEFAULT_NODE: Optional[WebSearchNode] = None


def _get_default_node() -> WebSearchNode:
    """
    Lazily constructs a default WebSearchNode instance.

    This keeps the previous simple functional API available while
    allowing full customization via the WebSearchNode class.
    """
    global _DEFAULT_NODE
    if _DEFAULT_NODE is None:
        _DEFAULT_NODE = WebSearchNode()
    return _DEFAULT_NODE


def websearch_node(state: WebSearchState) -> WebSearchState:
    """
    Functional, synchronous wrapper around the default WebSearchNode.

    Suitable for simple integrations where custom configuration is not required.
    """
    node = _get_default_node()
    return node.run(state)


async def websearch_node_async(state: WebSearchState) -> WebSearchState:
    """
    Functional, async wrapper around the default WebSearchNode.

    Suitable for LangGraph graphs and async environments.
    """
    node = _get_default_node()
    return await node.run_async(state)
