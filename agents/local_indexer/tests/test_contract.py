# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from local_indexer.local_indexer_agent import LocalIndexerAgent


@pytest.mark.unit
def test_local_indexer_contract_declares_local_workspace_index_skill() -> None:
    contract = LocalIndexerAgent().get_contract()
    assert "local.workspace.index" in contract.capabilities
    assert any(s.skill_id == "local.workspace.index" for s in contract.skills)


@pytest.mark.unit
def test_local_indexer_contract_resolves_rag_ingest_on_register() -> None:
    registry = AgentRegistry()
    registry.register(LocalIndexerAgent())
    contract = registry.get_contract("local_indexer")
    assert RAG_INGEST_TOOL_ID in contract.allowed_tools
