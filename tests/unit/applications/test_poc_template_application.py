# © Artur Czarnecki. All rights reserved.

"""Gate smoke for committed Phase N reference app (N.6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POC_ROOT = _REPO_ROOT / "applications" / "poc_template_application"


def test_poc_template_application_tree_exists() -> None:
    assert (_POC_ROOT / "manifest.py").is_file()
    assert (_POC_ROOT / "mcp" / "server.py").is_file()
    assert (_POC_ROOT / "docker" / "Dockerfile").is_file()
    assert (_POC_ROOT / "docker" / "build-docker.sh").is_file()
    assert (_POC_ROOT / "docker" / "build-docker.bat").is_file()
    assert (_POC_ROOT / "BUILD_AND_DEPLOY.md").is_file()
    assert (_POC_ROOT / "poc_template_application_tests" / "host").is_dir()


def test_poc_template_application_smoke() -> None:
    from poc_template_application.host.factory import create_poc_template_application
    from poc_template_application.host.settings import PocTemplateApplicationSettings

    settings = PocTemplateApplicationSettings(
        include_mcp=False,
        include_scheduler=False,
        include_interaction_routes=False,
    )
    client = TestClient(create_poc_template_application(settings=settings))
    response = client.get("/v1/poc_template/agents")
    assert response.status_code == 200
    assert response.json()["agents"]
