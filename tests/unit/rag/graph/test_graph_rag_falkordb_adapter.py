# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from tests.unit.rag.graph.test_graph_rag_memgraph_adapter import _FakeMemgraphIntegration


@pytest.mark.gate
def test_falkordb_adapter_via_integration_contract() -> None:
    store = create_rag_graph_store(
        integration_graph_store=_FakeMemgraphIntegration(),
    )
    assert isinstance(store, CypherRagGraphStore)
