# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Resolve RAG GraphStore backend from profile / env via backend registry (M-RAG.38)."""

from __future__ import annotations

import os
from typing import Any, Optional

from intergrax.rag.graph.bootstrap.backend_registry import (
    create_graph_store_from_registry,
    list_graph_store_backends,
    register_graph_store_backend,
)
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile

_BACKENDS_REGISTERED = False


def _factory_inmemory(
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    _ = integration_graph_store
    return InMemoryGraphStore(tenant_id=tenant_id)


def _factory_neo4j(
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    store = integration_graph_store
    if store is None:
        from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store

        store = create_neo4j_graph_store()
    return Neo4jRagGraphStore(store, tenant_id=tenant_id)


def _factory_memgraph(
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    store = integration_graph_store
    if store is None:
        from intergrax.integrations.providers.graph_store.memgraph.bundle import create_memgraph_graph_store

        store = create_memgraph_graph_store()
    return CypherRagGraphStore(store, tenant_id=tenant_id)


def _factory_falkordb(
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    store = integration_graph_store
    if store is None:
        from intergrax.integrations.providers.graph_store.falkordb.bundle import create_falkordb_graph_store

        store = create_falkordb_graph_store()
    return CypherRagGraphStore(store, tenant_id=tenant_id)


def _factory_arangodb(
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    store = integration_graph_store
    if store is None:
        from intergrax.integrations.providers.graph_store.arangodb.bundle import create_arangodb_graph_store

        store = create_arangodb_graph_store()
    return CypherRagGraphStore(store, tenant_id=tenant_id)


def _factory_neptune(
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    store = integration_graph_store
    if store is None:
        from intergrax.integrations.providers.graph_store.neptune.bundle import create_neptune_graph_store

        store = create_neptune_graph_store()
    return CypherRagGraphStore(store, tenant_id=tenant_id)


def _factory_orientdb(
    *,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    store = integration_graph_store
    if store is None:
        from intergrax.integrations.providers.graph_store.orientdb.bundle import create_orientdb_graph_store

        store = create_orientdb_graph_store()
    return CypherRagGraphStore(store, tenant_id=tenant_id)


def ensure_graph_store_backends_registered() -> None:
    global _BACKENDS_REGISTERED
    if _BACKENDS_REGISTERED:
        return
    register_graph_store_backend("inmemory", _factory_inmemory)
    register_graph_store_backend("neo4j", _factory_neo4j)
    register_graph_store_backend("memgraph", _factory_memgraph)
    register_graph_store_backend("falkordb", _factory_falkordb)
    register_graph_store_backend("neptune", _factory_neptune)
    register_graph_store_backend("orientdb", _factory_orientdb)
    register_graph_store_backend("arangodb", _factory_arangodb)
    _BACKENDS_REGISTERED = True


def create_rag_graph_store(
    *,
    profile: Optional[RagProfile] = None,
    integration_graph_store: Any = None,
    tenant_id: str | None = None,
) -> GraphStore:
    """
    Build a GraphRAG store via ``RagGraphStoreBackend`` registry.

    ``INTERGRAX_RAG_GRAPH_STORE`` / ``RagProfile.graph_store_backend``:
    ``inmemory`` (default) · ``neo4j`` · ``memgraph`` · ``falkordb`` · ``neptune`` · ``orientdb`` · ``arangodb``.
    """
    ensure_graph_store_backends_registered()
    profile = profile or RagProfile()
    backend = (
        profile.graph_store_backend
        or os.getenv("INTERGRAX_RAG_GRAPH_STORE", "inmemory").strip().lower()
        or "inmemory"
    )
    if backend not in list_graph_store_backends():
        raise ValueError(
            f"unknown_rag_graph_store_backend:{backend}; "
            f"known={','.join(list_graph_store_backends())}"
        )
    return create_graph_store_from_registry(
        backend,
        integration_graph_store=integration_graph_store,
        tenant_id=tenant_id,
    )
