# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID
from local_search.local_search_agent import LocalSearchAgent


@pytest.mark.unit
def test_local_search_contract_resolves_rag_retrieve_on_register() -> None:
    registry = AgentRegistry()
    registry.register(LocalSearchAgent())
    contract = registry.get_contract("local_search")
    assert RAG_RETRIEVE_TOOL_ID in contract.allowed_tools


@pytest.mark.unit
def test_local_search_contract_declares_local_workspace_search_skill() -> None:
    contract = LocalSearchAgent().get_contract()
    assert "local.workspace.search" in contract.capabilities
    assert any(s.skill_id == "local.workspace.search" for s in contract.skills)
