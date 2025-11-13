# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any, Annotated

import asyncio  # <- add this import

try:
    from langgraph.graph.message import add_messages
except ImportError:
    def add_messages(x: Any) -> Any:
        return x

from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.pipeline.default_pipeline import build_default_pipeline
from intergrax.websearch.schemas.web_document import WebDocument


class WebSearchState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_question: Optional[str]
    websearch_query: Optional[str]
    websearch_docs: Optional[List[Dict[str, Any]]]


_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_default_pipeline()
    return _PIPELINE


def _serialize_web_document(doc: WebDocument, max_text_chars: int = 4000) -> Dict[str, Any]:
    page = doc.page
    hit = doc.hit

    full_text = page.text or ""
    if max_text_chars and len(full_text) > max_text_chars:
        text = full_text[:max_text_chars]
    else:
        text = full_text

    return {
        "provider": hit.provider,
        "rank": hit.rank,
        "source_rank": doc.source_rank,
        "quality_score": doc.quality_score,
        "title": page.title or hit.title,
        "url": hit.url,
        "snippet": hit.snippet,
        "description": page.description,
        "lang": page.lang,
        "domain": hit.domain(),
        "published_at": hit.published_at.isoformat() if hit.published_at else None,
        "fetched_at": page.fetched_at.isoformat(),
        "text": text,
    }


def websearch_node(state: WebSearchState) -> WebSearchState:
    """
    Synchronous node – suitable for non-async environments.

    In environments with a running event loop (e.g. Jupyter),
    prefer using 'websearch_node_async' directly.
    """
    query = (state.get("websearch_query") or state.get("user_question") or "").strip()
    if not query:
        state["websearch_docs"] = []
        return state

    spec = QuerySpec(
        query=query,
        top_k=8,
        locale="pl-PL",
        region="pl-PL",
        language="pl",
        safe_search=True,
    )

    pipeline = _get_pipeline()
    docs = pipeline.run_sync(spec, top_n_fetch=8)

    state["websearch_docs"] = [
        _serialize_web_document(d) for d in docs
    ]
    return state


async def websearch_node_async(state: WebSearchState) -> WebSearchState:
    """
    Async version of the websearch node.

    This version is safe to use in environments where an event loop is
    already running (e.g. Jupyter notebooks, async web frameworks, LangGraph).
    """
    query = (state.get("websearch_query") or state.get("user_question") or "").strip()
    if not query:
        state["websearch_docs"] = []
        return state

    spec = QuerySpec(
        query=query,
        top_k=8,
        locale="pl-PL",
        region="pl-PL",
        language="pl",
        safe_search=True,
    )

    pipeline = _get_pipeline()
    docs = await pipeline.run(spec, top_n_fetch=8)

    state["websearch_docs"] = [
        _serialize_web_document(d) for d in docs
    ]
    return state
