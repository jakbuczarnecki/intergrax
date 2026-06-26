# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
APPLICATIONS = (
    "lab_application",
    "legal_application",
    "local_workspace_application",
    "poc_template_application",
    "research_application",
    "attestation_demo",
)


@pytest.mark.gate
@pytest.mark.parametrize("app_pkg", APPLICATIONS)
def test_application_deploy_triad_present(app_pkg: str) -> None:
    root = REPO / "applications" / app_pkg
    assert (root / "docker" / "Dockerfile").is_file(), f"{app_pkg}: missing docker/Dockerfile"
    assert (root / "docker" / "docker-compose.yml").is_file()
    deploy_doc = root / "docs" / "BUILD_AND_DEPLOY.md"
    if not deploy_doc.is_file() and app_pkg == "attestation_demo":
        deploy_doc = root / "BUILD_AND_DEPLOY.md"
    assert deploy_doc.is_file()
    build_sh = root / "docker" / "build-docker.sh"
    build_bat = root / "docker" / "build-docker.bat"
    assert build_sh.is_file() or build_bat.is_file()
