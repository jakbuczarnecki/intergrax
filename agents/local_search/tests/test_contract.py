# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.registry.agent_registry import AgentRegistry
from lkw_shared.tool_catalog import RAG_RETRIEVE_TOOL_ID
from local_search.local_search_agent import LocalSearchAgent


@pytest.mark.unit
def test_local_search_contract_allows_rag_retrieve() -> None:
    registry = AgentRegistry()
    registry.register(LocalSearchAgent())
    contract = registry.get_contract("local_search")
    assert RAG_RETRIEVE_TOOL_ID in contract.allowed_tools
