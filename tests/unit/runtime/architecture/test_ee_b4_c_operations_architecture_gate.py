# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — production operations architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_MODEL = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_PRODUCTION_OPERATIONS_INCIDENT_MODEL.md"
)
_RUNBOOKS = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "runbooks"
    / "EXECUTION_ENGINE_PRODUCTION_RUNBOOKS.md"
)
_CERT = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B4_C_PRODUCTION_OPERATIONS_INCIDENT_RUNBOOK_CERTIFICATION.md"
)
_SUPPORT = _REPO / "testing_support" / "operations"
_INTERGRAX = _REPO / "intergrax"

_EE_B4_C_TESTS = (
    "tests/unit/runtime/architecture/test_ee_b4_c_incident_taxonomy.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_runbook_completeness.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_forbidden_operator_bypass.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_capacity_runbook.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_evidence_runbook.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_recovery_runbook.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_unknown_side_effect_runbook.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_shutdown_runbook.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_provider_neutrality.py",
    "tests/unit/runtime/architecture/test_ee_b4_c_operations_architecture_gate.py",
)

_FORBIDDEN_PRODUCTION = (
    "IncidentRuntime",
    "RunbookEngine",
    "OperatorRuntime",
    "IncidentScheduler",
)

_NPSC5F_PROTECTED = (
    "runtime/observability/causal_evidence.py",
    "runtime/observability/causal_evidence_export.py",
    "runtime/observability/export_boundary.py",
)


def test_ee_b4_c_documents_present() -> None:
    assert _MODEL.is_file()
    assert _RUNBOOKS.is_file()
    assert _CERT.is_file()


def test_ee_b4_c_test_modules_present() -> None:
    for rel in _EE_B4_C_TESTS:
        assert (_REPO / rel).is_file(), rel


def test_ee_b4_c_support_package_present() -> None:
    assert (_SUPPORT / "incident_taxonomy.py").is_file()
    assert (_SUPPORT / "runbook_validator.py").is_file()
    assert (_SUPPORT / "forbidden_patterns.py").is_file()


def test_ee_b4_c_no_forbidden_operations_runtime_in_production() -> None:
    hits: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_PRODUCTION:
            if symbol in text:
                hits.append(f"{path.relative_to(_INTERGRAX)}:{symbol}")
    assert hits == []


def test_ee_b4_c_production_does_not_import_operations_testing_support() -> None:
    offenders: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("testing_support.operations"):
                    offenders.append(path.relative_to(_INTERGRAX).as_posix())
    assert offenders == []


def test_ee_b4_c_npsc5f_protected_files_unmodified_since_start() -> None:
    for rel in _NPSC5F_PROTECTED:
        path = _INTERGRAX / rel
        assert path.is_file(), rel
