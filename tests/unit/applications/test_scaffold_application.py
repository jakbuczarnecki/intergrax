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
    assert (target / "docs" / "ARCHITECTURE.md").exists()
    assert (target / "docs" / "IMPLEMENTATION_PLAN.md").exists()
    assert (target / ".env.example").exists()
    assert (target / "host" / "factory.py").exists()
    assert (target / "host" / "wiring.py").exists()
    assert (target / "host" / "agent_builders.py").exists()
    assert (target / "serving" / "fastapi_router.py").exists()
    assert (target / "mcp" / "server.py").exists()

    mcp_src = (target / "mcp" / "server.py").read_text(encoding="utf-8")
    assert "FastMCP" in mcp_src
    assert "build_concept_lab_mcp_server" in mcp_src
    factory_src = (target / "host" / "factory.py").read_text(encoding="utf-8")
    assert "couple_fastapi_with_mcp" in factory_src
    assert "build_concept_lab_mcp_server" in factory_src
    assert (target / "tests" / "host" / "test_concept_lab_host_smoke.py").exists()
    assert (target / "docker" / "Dockerfile").exists()
    assert (target / "docker" / ".dockerignore").exists()
    assert (target / "docker" / "docker-compose.yml").exists()
    assert (target / "docker" / "build-docker.sh").exists()
    assert (target / "scripts" / "build-local-docker.sh").exists()
    assert (target / "scripts" / "build-local-docker.bat").exists()
    assert (target / "docker" / "build-docker.bat").exists()
    assert (target / "sample_docs" / ".gitignore").exists()
    assert (target / "docs" / "BUILD_AND_DEPLOY.md").exists()

    deploy_doc = (target / "docs" / "BUILD_AND_DEPLOY.md").read_text(encoding="utf-8")
    assert f"applications/{pkg}/docker/Dockerfile" in deploy_doc
    assert f"applications/{pkg}/docker/build-docker.sh" in deploy_doc
    assert "CONCEPT_LAB_" in deploy_doc

    dockerfile = (target / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert f"applications/{pkg}/" in dockerfile
    assert "COPY agents/echo/" in dockerfile
    assert f"{pkg}.host.main:app" in dockerfile

    dockerignore = (target / "docker" / ".dockerignore").read_text(encoding="utf-8")
    assert f"!applications/{pkg}/" in dockerignore
    assert "!agents/echo/" in dockerignore

    manifest = (target / "manifest.py").read_text(encoding="utf-8")
    assert "AgentBinding.mount(EchoAgent" in manifest
    assert "build_application_registry" in (target / "host" / "wiring.py").read_text(encoding="utf-8")
    env_example = (target / ".env.example").read_text(encoding="utf-8")
    assert "CONCEPT_LAB_BACKEND_PORT=8092" in env_example
    assert "app_id=\"concept_lab\"" in (target / "manifest.py").read_text(encoding="utf-8")
    assert "CONCEPT_LAB_AGENT_BUILDERS" in (target / "host" / "wiring.py").read_text(encoding="utf-8")

