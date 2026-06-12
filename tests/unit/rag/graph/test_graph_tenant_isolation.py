# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.graph.tenant.graph_isolation_contract import run_graph_isolation_contract


@pytest.mark.gate
def test_inmemory_graph_tenant_isolation_contract() -> None:
    result = run_graph_isolation_contract(
        lambda tenant: InMemoryGraphStore(tenant_id=tenant),
        slug="inmemory",
    )
    assert result.cross_query_isolated is True
    assert result.reason == "ok"
