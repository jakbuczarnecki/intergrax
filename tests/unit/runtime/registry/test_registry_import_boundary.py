# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PACKAGE = REPO_ROOT / "intergrax" / "runtime" / "registry"

_FORBIDDEN_FRAGMENTS = (
    "echo.echo_agent",
    "EchoAgent",
    "research.research_agent",
    "ResearchAgent",
    "research.summary_agent",
    "SummaryAgent",
    "legal.legal_agent",
    "LegalAgent",
    "organization_worker.organization_worker_agent",
    "OrganizationWorkerAgent",
    "build_harness_registry",
    "build_research_registry",
    "build_legal_registry",
    "build_organization_worker_registry",
    "importlib.import_module",
)


def test_runtime_registry_package_has_no_concrete_tier2_agent_knowledge() -> None:
    violations: list[str] = []
    for path in sorted(REGISTRY_PACKAGE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for fragment in _FORBIDDEN_FRAGMENTS:
            if fragment in text:
                violations.append(f"{rel}: contains forbidden fragment {fragment!r}")
    assert not violations, "\n".join(violations)


def test_runtime_registry_package_has_no_bootstrap_module() -> None:
    bootstrap = REGISTRY_PACKAGE / "bootstrap.py"
    assert not bootstrap.exists(), "runtime/registry/bootstrap.py must not exist"
