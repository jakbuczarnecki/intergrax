# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.scaffold.new_agent import create_agent

pytestmark = [pytest.mark.unit, pytest.mark.agent_os, pytest.mark.gate]


def test_scaffold_creates_typed_reflex_agent_tree(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "agents").mkdir()

    target = create_agent(
        name="document_automation",
        capabilities=["documents.automation"],
        root=root,
    )

    assert target.is_dir()
    assert (target / "document_automation_agent.py").exists()
    assert (target / "contract.py").exists()
    assert (target / "capabilities.py").exists()
    assert not (target / "steps" / "pipeline.py").exists()
    assert (target / "prompts" / "system.md").exists()
    assert (target / "schemas" / "__init__.py").exists()
    assert (target / "tests" / "test_document_automation_agent.py").exists()
    assert (target / "README.md").exists()

    agent_py = (target / "document_automation_agent.py").read_text(encoding="utf-8")
    assert "class DocumentAutomationAgent(ReflexAgent)" in agent_py
    assert "async def perceive" in agent_py
    assert "def get_steps" not in agent_py

    content = (target / "capabilities.py").read_text(encoding="utf-8")
    assert "documents.automation" in content


def test_scaffold_uaep_flag_creates_legacy_pipeline(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "agents").mkdir()

    target = create_agent(
        name="legacy_agent",
        capabilities=["legacy.basic"],
        root=root,
        uaep=True,
    )

    assert (target / "steps" / "pipeline.py").exists()
    agent_py = (target / "legacy_agent_agent.py").read_text(encoding="utf-8")
    assert "def get_steps" in agent_py
