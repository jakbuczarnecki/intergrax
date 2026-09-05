# © Artur Czarnecki. All rights reserved.

import pytest

from local_indexer.local_indexer_agent import LocalIndexerAgent


@pytest.mark.unit
def test_local_indexer_contract_declares_local_workspace_index_skill() -> None:
    contract = LocalIndexerAgent().get_contract()
    assert "local.workspace.index" in contract.capabilities
    assert any(s.skill_id == "local.workspace.index" for s in contract.skills)
