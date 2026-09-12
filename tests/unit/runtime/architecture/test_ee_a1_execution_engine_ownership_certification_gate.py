# © Artur Czarnecki. All rights reserved.

"""EE-A1 — Execution Engine ownership & boundary certification gate."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OWNERSHIP_MODEL = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_OWNERSHIP_MODEL.md"
)
_P0_INVENTORY = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
)

_REQUIRED_SECTIONS = (
    "## 1. Execution Engine purpose",
    "## 2. Canonical execution flow",
    "## 3. Ownership model",
    "## 4. Forbidden bypasses",
    "## 5. Extension rules",
    "## 6. Plugin rules",
    "## 7. Recovery rules",
    "## 8. Evidence rules",
)

_FROZEN_PROOF_GATES = (
    "tests/unit/runtime/architecture/test_platform_execution_unification_p0_bypass_inventory.py",
    "tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py",
    "tests/unit/runtime/architecture/test_npsc3c_d_canonical_execution_engine_conformance_gate.py",
    "tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py",
)


def test_ee_a1_ownership_model_document_present_and_structured() -> None:
    assert _OWNERSHIP_MODEL.is_file(), "EXECUTION_ENGINE_OWNERSHIP_MODEL.md missing"
    text = _OWNERSHIP_MODEL.read_text(encoding="utf-8")
    for heading in _REQUIRED_SECTIONS:
        assert heading in text, f"missing section {heading!r}"
    assert "ExecutionRuntime" in text
    assert "RuntimeToolInvoker" in text
    assert "LongRunningCoordinator" in text
    assert "ExecutionAttemptRetryService" in text
    assert "FanOutPartialRecoveryService" in text
    assert "RuntimeEventPersistence" in text


def test_ee_a1_frozen_proof_gates_exist() -> None:
    missing = [rel for rel in _FROZEN_PROOF_GATES if not (_REPO_ROOT / rel).is_file()]
    assert missing == [], f"frozen proof gates missing: {missing}"


def test_ee_a1_p0_inventory_reports_zero_production_bypass() -> None:
    """EE-A1 requires BYPASS=0 in frozen P0 central inventory metrics."""
    text = _P0_INVENTORY.read_text(encoding="utf-8")
    match = re.search(r"^\| BYPASS \| (\d+) \|", text, flags=re.MULTILINE)
    assert match is not None, "P0 inventory missing BYPASS metric row"
    assert int(match.group(1)) == 0
    supported = re.search(
        r"^\| Supported execution bypasses \(production\) \| (\d+) \|",
        text,
        flags=re.MULTILINE,
    )
    assert supported is not None
    assert int(supported.group(1)) == 0
