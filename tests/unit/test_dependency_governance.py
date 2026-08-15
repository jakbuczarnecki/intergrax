from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

from scripts.maintenance.check_dependency_ownership import (
    classify_version_policy,
    check_project,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_current_project_passes_dependency_governance() -> None:
    assert check_project(ROOT / "pyproject.toml") == []


@pytest.mark.parametrize(
    ("specification", "expected"),
    (
        ("numpy==1.26.4", "EXACT_PIN"),
        ("fastapi>=0.115,<1", "BOUNDED_MAJOR"),
        ("pydantic>=2.7", "UNBOUNDED_MAJOR"),
    ),
)
def test_version_policy_classification(specification: str, expected: str) -> None:
    assert classify_version_policy(Requirement(specification)) == expected


def test_gate_rejects_core_capability_and_transitive_ownership(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[project]\n"
        "dependencies = ['fastmcp>=3.3.1', 'mcp>=1.0']\n"
        "[project.optional-dependencies]\n"
        "llm-all = ['langgraph>=1.0']\n",
        encoding="utf-8",
    )

    findings = check_project(path)

    assert "FORBIDDEN_CORE_PACKAGE: fastmcp" in findings
    assert "FORBIDDEN_CORE_PACKAGE: mcp" in findings
    assert "PROHIBITED_DIRECT_TRANSITIVE: core: mcp" in findings
    assert "LLM_ALL_LANGCHAIN_LEAK: langgraph" in findings
    assert any(finding.startswith("UNBOUNDED_MAJOR:") for finding in findings)


def test_gate_rejects_unapproved_core_name(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[project]\n"
        "dependencies = ['requests>=2']\n"
        "[project.optional-dependencies]\n",
        encoding="utf-8",
    )

    assert check_project(path) == ["CORE_OWNERSHIP_VIOLATION: requests"]
