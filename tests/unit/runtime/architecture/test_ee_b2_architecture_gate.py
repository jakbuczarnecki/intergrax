# © Artur Czarnecki. All rights reserved.

"""EE-B2 — chaos qualification architecture gates."""

from __future__ import annotations

import ast
import re
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
    / "EXECUTION_ENGINE_CHAOS_ENGINEERING_AND_FAULT_INJECTION_MODEL.md"
)
_CERT = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B2_CHAOS_ENGINEERING_FAULT_INJECTION_CERTIFICATION.md"
)
_CHAOS_SUPPORT = _REPO / "testing_support" / "chaos"
_INTERGRAX = _REPO / "intergrax"

_EE_B2_TESTS = (
    "tests/unit/runtime/architecture/test_ee_b2_worker_fault_injection.py",
    "tests/unit/runtime/architecture/test_ee_b2_dependency_fault_injection.py",
    "tests/unit/runtime/architecture/test_ee_b2_capacity_saturation_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_cancellation_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_evidence_persistence_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_checkpoint_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_recovery_interruption.py",
    "tests/unit/runtime/architecture/test_ee_b2_child_partial_failure.py",
    "tests/unit/runtime/architecture/test_ee_b2_observability_export_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_compound_failure.py",
    "tests/unit/runtime/architecture/test_ee_b2_shutdown_fault.py",
    "tests/unit/runtime/architecture/test_ee_b2_architecture_gate.py",
)

_FORBIDDEN_PRODUCTION = (
    "ChaosRuntime",
    "FailureRuntime",
    "ChaosScheduler",
    "ChaosWorkerManager",
    "FaultRecoveryEngine",
    "ChaosRetryEngine",
    "ChaosRecoveryEngine",
)

_REFLECTION_IN_CHAOS = re.compile(r"\b(getattr|setattr|hasattr)\s*\(")


def test_ee_b2_documents_present() -> None:
    assert _MODEL.is_file()
    assert _CERT.is_file()


def test_ee_b2_test_modules_present() -> None:
    for rel in _EE_B2_TESTS:
        assert (_REPO / rel).is_file(), rel


def test_ee_b2_chaos_support_package_present() -> None:
    assert (_CHAOS_SUPPORT / "fault_plan.py").is_file()
    assert (_CHAOS_SUPPORT / "__init__.py").is_file()


def test_ee_b2_no_forbidden_chaos_runtime_in_production() -> None:
    hits: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for sym in _FORBIDDEN_PRODUCTION:
            if sym in text:
                hits.append(f"{path.relative_to(_REPO)}:{sym}")
    assert hits == []


def test_ee_b2_production_does_not_import_testing_support_chaos() -> None:
    violations: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "testing_support.chaos" or alias.name.startswith(
                        "testing_support.chaos.",
                    ):
                        violations.append(path.relative_to(_REPO).as_posix())
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "testing_support.chaos" or node.module.startswith(
                    "testing_support.chaos.",
                ):
                    violations.append(path.relative_to(_REPO).as_posix())
    assert violations == []


def test_ee_b2_chaos_helpers_no_reflection_bypass() -> None:
    for path in _CHAOS_SUPPORT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert _REFLECTION_IN_CHAOS.search(text) is None, path.name


def test_ee_b2_chaos_helpers_no_identity_mint() -> None:
    mint_pattern = re.compile(
        r"\b(mint_run_id|mint_attempt_id|mint_execution_id|mint_task_id)\s*\(",
    )
    for path in _CHAOS_SUPPORT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert mint_pattern.search(text) is None, path.name
