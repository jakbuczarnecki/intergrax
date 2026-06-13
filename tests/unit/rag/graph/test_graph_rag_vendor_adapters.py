# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.bootstrap.backend_registry import list_graph_store_backends
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store, ensure_graph_store_backends_registered
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile
from tests.unit.rag.graph.fixtures.fake_cypher_graph_integration import FakeCypherGraphIntegration

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize("backend", ["neptune", "orientdb", "arangodb"])
def test_vendor_graph_rag_adapter_via_registry(backend: str) -> None:
    ensure_graph_store_backends_registered()
    assert backend in list_graph_store_backends()
    profile = RagProfile(graph_store_backend=backend, graph_rag_enabled=True)
    store = create_rag_graph_store(
        profile=profile,
        integration_graph_store=FakeCypherGraphIntegration(),
        tenant_id="vendor-graph",
    )
    assert isinstance(store, CypherRagGraphStore)
    assert store.tenant_id == "vendor-graph"
