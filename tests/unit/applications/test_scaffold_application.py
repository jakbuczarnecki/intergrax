# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.scaffold.new_application import create_application

pytestmark = [pytest.mark.unit, pytest.mark.agent_os, pytest.mark.gate]


def test_scaffold_creates_application_tree(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "applications").mkdir()

    target = create_application(
        name="concept_lab",
        agents=["echo"],
        profile="lab",
        root=root,
        port=8092,
        route_prefix="/v1/concept",
    )

    pkg = "concept_lab_application"
    assert target.is_dir()
    assert (target / "manifest.py").exists()
    assert (target / ".env.example").exists()
    assert (target / "host" / "factory.py").exists()
    assert (target / "host" / "wiring.py").exists()
    assert (target / "host" / "agent_builders.py").exists()
    assert (target / "serving" / "fastapi_router.py").exists()
    assert (target / f"{pkg}_tests" / "host" / "test_concept_lab_host_smoke.py").exists()

    manifest = (target / "manifest.py").read_text(encoding="utf-8")
    assert "AgentBinding.mount(EchoAgent" in manifest
    assert "build_application_registry" in (target / "host" / "wiring.py").read_text(encoding="utf-8")
    assert "CONCEPT_LAB_BACKEND_PORT=8092" in (target / ".env.example").read_text(encoding="utf-8")

