# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Resolve RAG GraphStore backend from profile / env (in-memory or Neo4j)."""

from __future__ import annotations

import os
from typing import Any, Optional

from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile


def create_rag_graph_store(
    *,
    profile: Optional[RagProfile] = None,
    integration_graph_store: Any = None,
) -> GraphStore:
    """
    Build a GraphRAG store.

    ``INTERGRAX_RAG_GRAPH_STORE``:
    - ``inmemory`` (default)
    - ``neo4j`` — wraps Integration Library ``create_neo4j_graph_store()``
    """
    profile = profile or RagProfile()
    backend = (
        profile.graph_store_backend
        or os.getenv("INTERGRAX_RAG_GRAPH_STORE", "inmemory").strip().lower()
        or "inmemory"
    )
    if backend == "neo4j":
        store = integration_graph_store
        if store is None:
            from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store

            store = create_neo4j_graph_store()
        return Neo4jRagGraphStore(store)
    return InMemoryGraphStore()
