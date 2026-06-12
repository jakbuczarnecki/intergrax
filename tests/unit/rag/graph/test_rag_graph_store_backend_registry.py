# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.bootstrap.backend_registry import (
    list_graph_store_backends,
    register_graph_store_backend,
    resolve_graph_store_backend,
)
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import (
    create_rag_graph_store,
    ensure_graph_store_backends_registered,
)
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile


@pytest.mark.gate
def test_shipped_graph_store_backends_registered() -> None:
    ensure_graph_store_backends_registered()
    backends = list_graph_store_backends()
    assert "inmemory" in backends
    assert "neo4j" in backends
    assert "memgraph" in backends
    assert "falkordb" in backends


@pytest.mark.gate
def test_create_rag_graph_store_resolves_inmemory() -> None:
    profile = RagProfile(graph_store_backend="inmemory", graph_rag_enabled=True)
    store = create_rag_graph_store(profile=profile)
    assert isinstance(store, InMemoryGraphStore)


@pytest.mark.gate
def test_register_custom_graph_store_backend() -> None:
    def _custom_factory(**_kwargs):
        return InMemoryGraphStore(tenant_id="custom")

    register_graph_store_backend("custom_lab", _custom_factory)
    factory = resolve_graph_store_backend("custom_lab")
    assert factory is not None
    store = factory()
    assert store.tenant_id == "custom"
