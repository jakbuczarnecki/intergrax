# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intergrax.scaffold.agent_catalog import resolve_agent_specs
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.new_agent import create_agent
from intergrax.scaffold.new_application import create_application


@pytest.mark.gate
def test_minimal_stack_scaffold_and_http_run_under_60s() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        slug = "dx_minimal_stack"
        started = time.monotonic()
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
        pkg = names.pkg
        import importlib
        import sys

        sys.path.insert(0, str(root / "agents"))
        sys.path.insert(0, str(root / "applications"))
        factory_mod = importlib.import_module(f"{pkg}.host.factory")
        factory_name = f"create_{names.short}_application"
        app = factory_mod.__dict__[factory_name]()
        client = TestClient(app)
        agents = client.get(f"{names.route_prefix}/agents")
        assert agents.status_code == 200
        run = client.post(
            f"{names.route_prefix}/run",
            json={"message": "dx minimal", "capability": f"{slug}.basic"},
        )
        assert run.status_code == 200
        assert run.json().get("state") == "completed"
        elapsed = time.monotonic() - started
        assert elapsed < 60.0
