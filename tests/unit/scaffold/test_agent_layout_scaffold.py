# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intergrax.scaffold.new_agent import create_agent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_scaffolded_agent_docs_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = create_agent(
            name="layout_probe",
            capabilities=["layout_probe.basic"],
            root=root,
            force=True,
            minimal=True,
        )
        docs = target / "docs"
        assert (docs / "ARCHITECTURE.md").is_file()
        assert (docs / "IMPLEMENTATION_PLAN.md").is_file()
        assert (docs / "adr" / "README.md").is_file()
        assert (docs / "journal" / ".gitkeep").is_file()
        assert not (target / "ARCHITECTURE.md").exists()
        assert not (target / "IMPLEMENTATION_PLAN.md").exists()
        assert not (target / "adr").exists()

        readme = (target / "README.md").read_text(encoding="utf-8")
        assert "docs/ARCHITECTURE.md" in readme
        assert "docs/IMPLEMENTATION_PLAN.md" in readme
        assert "docs/adr/README.md" in readme
        assert "`ARCHITECTURE.md`](ARCHITECTURE.md)" not in readme
        assert "uv run pytest agents/layout_probe/tests -q" in readme
