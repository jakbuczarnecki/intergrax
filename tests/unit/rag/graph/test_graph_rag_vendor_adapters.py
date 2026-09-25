# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from tests.unit.rag.graph.fixtures.fake_cypher_graph_integration import FakeCypherGraphIntegration

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize("tenant_id", ["vendor-neptune", "vendor-orient", "vendor-arango"])
def test_cypher_adapter_wraps_integration_graph_store(tenant_id: str) -> None:
    store = create_rag_graph_store(
        integration_graph_store=FakeCypherGraphIntegration(),
        tenant_id=tenant_id,
    )
    assert isinstance(store, CypherRagGraphStore)
    assert store.tenant_id == tenant_id
