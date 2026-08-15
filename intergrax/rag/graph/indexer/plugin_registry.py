# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GraphIndexer plugin registry (M-RAG.46)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.profiles.rag_profile import RagProfile


class GraphIndexerPlugin(Protocol):
    def index_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        *,
        chunk_ids: Sequence[str] | None = None,
    ) -> int: ...


GraphIndexerFactory = Callable[[GraphStore, RagProfile, LLMAdapter | None], GraphIndexerPlugin]

_REGISTRY: dict[str, GraphIndexerFactory] = {}


def register_graph_indexer_plugin(plugin_id: str, factory: GraphIndexerFactory) -> None:
    key = plugin_id.strip().lower()
    if not key:
        raise ValueError("plugin_id must be non-empty")
    _REGISTRY[key] = factory


def resolve_graph_indexer_plugin(plugin_id: str) -> GraphIndexerFactory | None:
    return _REGISTRY.get(plugin_id.strip().lower())


def list_graph_indexer_plugins() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))
