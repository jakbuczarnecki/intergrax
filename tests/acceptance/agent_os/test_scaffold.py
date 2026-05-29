# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.scaffold.new_agent import create_agent

pytestmark = [pytest.mark.unit, pytest.mark.agent_os, pytest.mark.gate]


def test_scaffold_creates_uaep_agent_tree(tmp_path):
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
    assert (target / "steps" / "pipeline.py").exists()
    assert (target / "prompts" / "system.md").exists()
    assert (target / "schemas" / "__init__.py").exists()
    assert (target / "tests" / "test_document_automation_agent.py").exists()
    assert (target / "notebooks" / "01_document_automation_experiment.ipynb").exists()
    assert (target / "README.md").exists()

    content = (target / "capabilities.py").read_text(encoding="utf-8")
    assert "documents.automation" in content
