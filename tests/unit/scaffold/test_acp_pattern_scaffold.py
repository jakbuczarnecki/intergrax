# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.scaffold.new_agent import create_agent


@pytest.mark.unit
@pytest.mark.gate
def test_scaffold_react_pattern_emits_typed_agent(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "agents").mkdir()

    target = create_agent(
        name="analyst",
        capabilities=["research.deep"],
        root=root,
        pattern="react",
    )

    agent_py = (target / "analyst_agent.py").read_text(encoding="utf-8")
    contract_py = (target / "contract.py").read_text(encoding="utf-8")

    assert "class AnalystAgent(ReActAgent)" in agent_py
    assert "async def perceive" in agent_py
    assert "def get_steps" not in agent_py
    assert "async def run_step" not in agent_py
    assert "CognitivePattern.REACT" in contract_py
    assert "cognitive_pattern=_PATTERN" in contract_py
