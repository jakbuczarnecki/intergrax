# © Artur Czarnecki. All rights reserved.

import pytest

from local_search.local_search_agent import LocalSearchAgent


@pytest.mark.unit
def test_local_search_contract_declares_local_workspace_search_skill() -> None:
    contract = LocalSearchAgent().get_contract()
    assert "local.workspace.search" in contract.capabilities
    assert any(s.skill_id == "local.workspace.search" for s in contract.skills)
