# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.no_ci]

from intergrax.scaffold.agent_catalog import resolve_agent_specs
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.new_application import _create_lab_application


@pytest.mark.gate
def test_scaffolded_lab_application_includes_deploy_triad() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specs = resolve_agent_specs(["echo"])
        names = ScaffoldApplicationNames.resolve("scaffold_triad_test")
        target = root / "applications" / names.pkg
        _create_lab_application(
            names=names,
            specs=specs,
            target=target,
            profile="lab",
            force=True,
            full_scaffold=False,
        )
        assert (target / "docker" / "Dockerfile").is_file()
        assert (target / "docker" / "docker-compose.yml").is_file()
        assert (target / "BUILD_AND_DEPLOY.md").is_file()
        assert (target / "ARCHITECTURE.md").is_file()
        assert (target / "IMPLEMENTATION_PLAN.md").is_file()
        assert (target / "host" / "factory.py").read_text(encoding="utf-8").count(
            "build_harness_host_runtime"
        ) >= 1
