# © Artur Czarnecki. All rights reserved.

"""Gate: ADR scaffold (README + TEMPLATE) for Harness, agents, and applications."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from intergrax.scaffold.adr_templates import (
    ADR_README,
    ADR_TEMPLATE,
    write_agent_adr_scaffold,
    write_application_adr_scaffold,
    write_harness_adr_scaffold,
)
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.new_agent import create_agent
from intergrax.scaffold.new_application import create_application

REPO = Path(__file__).resolve().parents[3]

AGENTS_WITH_ADR = (
    "boundary_demo",
    "echo",
    "research",
    "signoff_probe",
    "legal",
    "problem_radar",
    "organization_worker",
    "local_indexer",
    "local_search",
    "local_synthesizer",
)

APPLICATIONS_WITH_ADR = (
    "lab_application",
    "legal_application",
    "poc_template_application",
    "research_application",
    "local_workspace_application",
    "attestation_demo",
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _assert_adr_scaffold(adr_dir: Path, *, label: str) -> None:
    readme = adr_dir / ADR_README
    template = adr_dir / ADR_TEMPLATE
    assert readme.is_file(), f"{label}: missing adr/README.md"
    assert template.is_file(), f"{label}: missing adr/TEMPLATE.md"
    readme_text = readme.read_text(encoding="utf-8")
    template_text = template.read_text(encoding="utf-8")
    assert "TEMPLATE.md" in readme_text, f"{label}: README must link TEMPLATE.md"
    assert "Context" in template_text and "Decision" in template_text, f"{label}: invalid TEMPLATE.md"


@pytest.mark.parametrize("agent_slug", AGENTS_WITH_ADR)
def test_agent_adr_scaffold_present(agent_slug: str) -> None:
    _assert_adr_scaffold(REPO / "agents" / agent_slug / "docs" / "adr", label=f"agents/{agent_slug}")


def _application_adr_dir(app_pkg: str) -> Path:
    return REPO / "applications" / app_pkg / "docs" / "adr"


@pytest.mark.parametrize("app_pkg", APPLICATIONS_WITH_ADR)
def test_application_adr_scaffold_present(app_pkg: str) -> None:
    adr_dir = _application_adr_dir(app_pkg)
    _assert_adr_scaffold(adr_dir, label=f"applications/{app_pkg}")


def test_harness_adr_scaffold_present() -> None:
    _assert_adr_scaffold(REPO / "docs" / "adr", label="docs/adr")
    readme = (REPO / "docs" / "adr" / ADR_README).read_text(encoding="utf-8")
    assert "ADR-FLOW-001" in readme
    assert "entries/" in readme
    assert (REPO / "docs" / "adr" / "entries").is_dir()


def test_agent_scaffold_emits_adr_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = create_agent(
            name="adr_probe",
            capabilities=["adr_probe.basic"],
            root=root,
            force=True,
            minimal=True,
        )
        _assert_adr_scaffold(target / "docs" / "adr", label="scaffold agent")


def test_application_scaffold_emits_adr_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "applications").mkdir()
        target = create_application(
            name="adr_app",
            agents=["echo"],
            profile="lab",
            root=root,
            force=True,
            minimal=True,
        )
        _assert_adr_scaffold(target / "docs" / "adr", label="scaffold application")


def test_adr_template_renderers_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = write_harness_adr_scaffold(root=root, force=True)
        text_before = (first / ADR_README).read_text(encoding="utf-8")
        write_harness_adr_scaffold(root=root, force=False)
        text_after = (first / ADR_README).read_text(encoding="utf-8")
        assert text_before == text_after

        names = ScaffoldApplicationNames.resolve("adr_idem")
        app_dir = root / "applications" / names.pkg
        app_dir.mkdir(parents=True)
        write_application_adr_scaffold(
            app_dir=app_dir,
            pkg=names.pkg,
            short=names.short,
            display=names.display,
            force=True,
        )
        tpl_before = (app_dir / "docs" / "adr" / ADR_TEMPLATE).read_text(encoding="utf-8")
        write_application_adr_scaffold(
            app_dir=app_dir,
            pkg=names.pkg,
            short=names.short,
            display=names.display,
            force=False,
        )
        tpl_after = (app_dir / "docs" / "adr" / ADR_TEMPLATE).read_text(encoding="utf-8")
        assert tpl_before == tpl_after
