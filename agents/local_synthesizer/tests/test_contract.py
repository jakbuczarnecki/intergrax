# © Artur Czarnecki. All rights reserved.

import pytest

from local_synthesizer.local_synthesizer_agent import LocalSynthesizerAgent


@pytest.mark.unit
def test_local_synthesizer_contract_declares_local_workspace_synthesize_skill() -> None:
    contract = LocalSynthesizerAgent().get_contract()
    assert "local.workspace.synthesize" in contract.capabilities
    assert any(s.skill_id == "local.workspace.synthesize" for s in contract.skills)
