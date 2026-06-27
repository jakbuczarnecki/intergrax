# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intergrax.scaffold.new_agent import create_agent
from intergrax.scaffold.new_application import create_application

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_agent_scaffold_emits_architecture_and_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = create_agent(
            name="doc_probe",
            capabilities=["doc_probe.basic"],
            root=root,
            force=True,
            minimal=True,
        )
        arch = (target / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        plan = (target / "docs" / "IMPLEMENTATION_PLAN.md").read_text(encoding="utf-8")
        assert "doc_probe agent — architecture" in arch
        assert "IMPLEMENTATION_PLAN.md" in arch
        assert "ARCHITECTURE.md" in plan
        assert "doc_probe agent — Implementation Plan" in plan
        assert "DOC_PROBE-1" in plan
        assert "ARCHITECTURE.md" in plan
        assert (target / "docs" / "adr" / "README.md").is_file()
        assert (target / "docs" / "adr" / "TEMPLATE.md").is_file()


def test_application_scaffold_emits_architecture_and_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "applications").mkdir()
        target = create_application(
            name="doc_app",
            agents=["echo"],
            profile="lab",
            root=root,
            force=True,
            minimal=True,
        )
        assert (target / "docs" / "ARCHITECTURE.md").is_file()
        assert (target / "docs" / "IMPLEMENTATION_PLAN.md").is_file()
        plan = (target / "docs" / "IMPLEMENTATION_PLAN.md").read_text(encoding="utf-8")
        assert "DOC_APP-4" in plan
        assert "expand" in plan.lower()
        assert (target / "docs" / "adr" / "README.md").is_file()
        assert (target / "docs" / "adr" / "TEMPLATE.md").is_file()
