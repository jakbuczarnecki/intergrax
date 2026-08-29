# © Artur Czarnecki. All rights reserved.

"""UE-8P3 — canonical docs freeze pluggable ExecutionAuthorityPolicy semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UEA = _REPO_ROOT / "docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md"
_NEXUS = _REPO_ROOT / "docs/project/architecture/NEXUS_EXECUTION_FLOW.md"


def test_uea_documents_pluggable_authority_policy_model() -> None:
    text = _UEA.read_text(encoding="utf-8")
    assert "ExecutionAuthorityPolicy" in text
    assert "DefaultStrictAuthorityPolicy" in text
    assert "mandatory authority checkpoint" in text.lower()
    assert "intergrax.execution_authority_policies" in text
    assert "UEA-INV-022" in text
    assert "fails closed" in text.lower()
    assert "resolve_execution_authority_policy_from_runtime_config" in text
    # default strict narrowing is not universal platform law
    assert "not a universal law" in text.lower() or "not a universal platform law" in text.lower()


def test_nexus_docs_authority_checkpoint_not_owned_by_nexus() -> None:
    text = _NEXUS.read_text(encoding="utf-8")
    assert "Pluggable child Execution authority" in text
    assert "ExecutionAuthorityPolicy" in text
    assert "DefaultStrictAuthorityPolicy" in text
    assert "mandatory authority checkpoint" in text.lower()
    assert "intergrax.execution_authority_policies" in text
    assert "NEXUS-INV-011" in text
    assert "does not" in text.lower() and "bypass" in text.lower()
    # Nexus must not own policy selection/algorithm
    nexus_section = text.split("### Pluggable child Execution authority", 1)[1].split(
        "## Nexus-specific invariants", 1
    )[0]
    assert "does **not** implement" in nexus_section or "does not" in nexus_section.lower()
    assert "load" in nexus_section.lower() or "select" in nexus_section.lower()
