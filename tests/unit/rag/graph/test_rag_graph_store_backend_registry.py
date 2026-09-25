# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from tests.unit.rag.graph.fixtures.fake_cypher_graph_integration import FakeCypherGraphIntegration

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_create_rag_graph_store_defaults_to_inmemory() -> None:
    store = create_rag_graph_store()
    assert isinstance(store, InMemoryGraphStore)


def test_create_rag_graph_store_adapts_integration_contract() -> None:
    store = create_rag_graph_store(
        integration_graph_store=FakeCypherGraphIntegration(),
        tenant_id="lab",
    )
    assert isinstance(store, CypherRagGraphStore)
    assert store.tenant_id == "lab"
