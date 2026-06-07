# © Artur Czarnecki. All rights reserved.

"""Gate: Tier-2/Tier-3 modules must ship ARCHITECTURE.md + IMPLEMENTATION_PLAN.md pair."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

AGENTS_WITH_DOC_PAIR = (
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

APPLICATIONS_WITH_DOC_PAIR = (
    "lab_application",
    "legal_application",
    "poc_template_application",
    "research_application",
    "local_workspace_application",
)


def _assert_doc_pair(root: Path, label: str) -> None:
    architecture = root / "ARCHITECTURE.md"
    plan = root / "IMPLEMENTATION_PLAN.md"
    assert architecture.is_file(), f"{label}: missing ARCHITECTURE.md"
    assert plan.is_file(), f"{label}: missing IMPLEMENTATION_PLAN.md"
    arch_text = architecture.read_text(encoding="utf-8")
    plan_text = plan.read_text(encoding="utf-8")
    assert "IMPLEMENTATION_PLAN.md" in arch_text, f"{label}: ARCHITECTURE.md must link IMPLEMENTATION_PLAN.md"
    assert "ARCHITECTURE.md" in plan_text, f"{label}: IMPLEMENTATION_PLAN.md must link ARCHITECTURE.md"


@pytest.mark.gate
@pytest.mark.parametrize("agent_slug", AGENTS_WITH_DOC_PAIR)
def test_agent_doc_pair_present(agent_slug: str) -> None:
    _assert_doc_pair(REPO / "agents" / agent_slug, f"agents/{agent_slug}")


@pytest.mark.gate
@pytest.mark.parametrize("app_pkg", APPLICATIONS_WITH_DOC_PAIR)
def test_application_doc_pair_present(app_pkg: str) -> None:
    _assert_doc_pair(REPO / "applications" / app_pkg, f"applications/{app_pkg}")
