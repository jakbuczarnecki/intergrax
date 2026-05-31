# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Resolve graph indexer implementation from profile (heuristic vs optional LLM)."""

from __future__ import annotations

from typing import Literal, Optional, Protocol, Sequence

from langchain_core.documents import Document

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.indexer.llm_graph_indexer import LlmGraphIndexer
from intergrax.rag.profiles.rag_profile import RagProfile

GraphIndexerMode = Literal["heuristic", "llm", "heuristic_then_llm"]


class GraphIndexer(Protocol):
    def index_documents(
        self, documents: Sequence[Document], *, chunk_ids: Sequence[str] | None = None
    ) -> int: ...


def resolve_graph_indexer(
    store: GraphStore,
    profile: RagProfile,
    *,
    llm: Optional[LLMAdapter] = None,
) -> GraphIndexer:
    mode: GraphIndexerMode = profile.graph_indexer_mode  # type: ignore[attr-defined]
    if mode == "llm" and llm is not None:
        return LlmGraphIndexer(store, llm)
    if mode == "heuristic_then_llm" and llm is not None:
        return _CompositeIndexer(
            HeuristicGraphIndexer(store),
            LlmGraphIndexer(store, llm),
        )
    return HeuristicGraphIndexer(store)


class _CompositeIndexer:
    def __init__(self, *indexers: GraphIndexer) -> None:
        self._indexers = indexers

    def index_documents(
        self, documents: Sequence[Document], *, chunk_ids: Sequence[str] | None = None
    ) -> int:
        total = 0
        for indexer in self._indexers:
            total += indexer.index_documents(documents, chunk_ids=chunk_ids)
        return total
