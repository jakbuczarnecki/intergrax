# © Artur Czarnecki. All rights reserved.

"""EE-B1.2 — architecture gates (single owner, no bypass, no vendor coupling)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_CAPACITY_DOC = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_CAPACITY_AND_BACKPRESSURE_MODEL.md"
)
_CERT_DOC = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B1_2_EXECUTION_CAPACITY_BACKPRESSURE_ENTERPRISE_CERTIFICATION.md"
)
_CONTRACT_PKG = _REPO / "intergrax" / "contracts" / "execution_capacity"
_RUNTIME_CAPACITY = _REPO / "intergrax" / "runtime" / "execution" / "capacity"

_FORBIDDEN_SCHEDULER_NAMES = (
    "BackpressureScheduler",
    "CapacityScheduler",
    "ExecutionQueueEngine",
    "SecondExecutionRuntime",
    "CapacityRuntime",
    "AlternativeWorkerPool",
    "ExecutionDispatcher",
)

_VENDOR_TOKENS = ("redis", "kafka", "celery", "kubernetes")

_REQUIRED_DOC_SECTIONS = (
    "## 2. Capacity owner",
    "## 4. Capacity vs budget",
    "## 5. Capacity vs governance",
    "## 6. Capacity vs retry",
    "## 7. Local vs distributed semantics",
    "## 8. Admission state machine",
    "## 13. Nested child execution",
    "## 12. Layering (fan-out & Nexus)",
)


def _py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


def test_ee_b1_2_architecture_documents_present() -> None:
    assert _CAPACITY_DOC.is_file()
    assert _CERT_DOC.is_file()
    text = _CAPACITY_DOC.read_text(encoding="utf-8")
    for heading in _REQUIRED_DOC_SECTIONS:
        assert heading in text, f"missing {heading}"


def test_ee_b1_2_no_forbidden_second_plane_symbols_in_capacity_packages() -> None:
    roots = (_CONTRACT_PKG, _RUNTIME_CAPACITY)
    hits: list[str] = []
    for root in roots:
        for path in _py_files(root):
            rel = path.relative_to(_REPO).as_posix()
            body = path.read_text(encoding="utf-8")
            for symbol in _FORBIDDEN_SCHEDULER_NAMES:
                if symbol in body:
                    hits.append(f"{rel}: {symbol}")
    assert hits == []


def test_ee_b1_2_capacity_contract_no_governance_imports() -> None:
    for path in _py_files(_CONTRACT_PKG):
        text = path.read_text(encoding="utf-8")
        assert "PolicyEngine" not in text
        assert "GovernanceEngine" not in text


def test_ee_b1_2_capacity_contract_no_identity_mint() -> None:
    mint_pattern = re.compile(
        r"\b(mint_run_id|mint_attempt_id|mint_execution_id|mint_task_id)\s*\(",
    )
    for path in _py_files(_CONTRACT_PKG):
        text = path.read_text(encoding="utf-8")
        assert mint_pattern.search(text) is None, path.name


def test_ee_b1_2_capacity_contract_no_vendor_coupling() -> None:
    for path in _py_files(_CONTRACT_PKG):
        lower = path.read_text(encoding="utf-8").lower()
        for token in _VENDOR_TOKENS:
            assert token not in lower, f"{path.name} mentions {token}"


def test_ee_b1_2_gate_modules_exist() -> None:
    gates = (
        "tests/unit/runtime/architecture/test_ee_b1_2_capacity_contract.py",
        "tests/unit/runtime/architecture/test_ee_b1_2_capacity_admission.py",
        "tests/unit/runtime/architecture/test_ee_b1_2_capacity_concurrency.py",
        "tests/unit/runtime/architecture/test_ee_b1_2_capacity_release_semantics.py",
        "tests/unit/runtime/architecture/test_ee_b1_2_capacity_child_execution_interaction.py",
    )
    missing = [g for g in gates if not (_REPO / g).is_file()]
    assert missing == []
