# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intergrax.scaffold.agent_catalog import resolve_agent_specs
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.new_agent import create_agent
from intergrax.scaffold.new_application import create_application


@pytest.mark.gate
def test_minimal_stack_omits_docker_and_mcp() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        slug = "min_stack"
        create_agent(name=slug, capabilities=[f"{slug}.basic"], root=root, minimal=True, force=True)
        resolve_agent_specs([slug])
        create_application(
            name=slug,
            agents=[slug],
            profile="lab",
            root=root,
            force=True,
            minimal=True,
        )
        names = ScaffoldApplicationNames.resolve(slug)
        app_dir = root / "applications" / names.pkg
        assert not (app_dir / "docker").exists()
        assert not (app_dir / "mcp").exists()
        assert not (app_dir / "BUILD_AND_DEPLOY.md").is_file()
        assert (app_dir / "host" / "factory.py").is_file()
        assert "create_lab_fastapi_from_runtime" in (app_dir / "host" / "factory.py").read_text(encoding="utf-8")
        assert (app_dir / "package.json").is_file()
