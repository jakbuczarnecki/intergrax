# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intergrax.scaffold.new_application import _create_lab_application
from intergrax.scaffold.agent_catalog import resolve_agent_specs
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.application_layout import SAMPLE_DOCS_GITIGNORE

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_scaffolded_application_docs_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        specs = resolve_agent_specs(["echo"])
        names = ScaffoldApplicationNames.resolve("layout_probe")
        target = root / "applications" / names.pkg
        _create_lab_application(
            names=names,
            specs=specs,
            target=target,
            profile="lab",
            force=True,
            full_scaffold=False,
        )
        docs = target / "docs"
        assert (docs / "ARCHITECTURE.md").is_file()
        assert (docs / "IMPLEMENTATION_PLAN.md").is_file()
        assert (docs / "BUILD_AND_DEPLOY.md").is_file()
        assert (docs / "adr" / "README.md").is_file()
        assert (docs / "journal" / ".gitkeep").is_file()
        assert not (target / "ARCHITECTURE.md").exists()
        assert not (target / "IMPLEMENTATION_PLAN.md").exists()
        assert not (target / "BUILD_AND_DEPLOY.md").exists()
        assert not (target / "adr").exists()

        readme = (target / "README.md").read_text(encoding="utf-8")
        assert "docs/ARCHITECTURE.md" in readme
        assert "docs/IMPLEMENTATION_PLAN.md" in readme
        assert "docs/BUILD_AND_DEPLOY.md" in readme
        assert "docs/project/technical/adr/README.md" in readme
        assert "`ARCHITECTURE.md`](ARCHITECTURE.md)" not in readme

        assert (target / "scripts" / "build-local-docker.sh").is_file()
        assert (target / "scripts" / "build-local-docker.bat").is_file()
        assert not (target / "build-local-docker.sh").exists()

        gitignore = (target / "sample_docs" / ".gitignore").read_text(encoding="utf-8")
        assert gitignore.strip() == SAMPLE_DOCS_GITIGNORE.strip()
        assert (target / "sample_docs" / ".gitkeep").is_file()
