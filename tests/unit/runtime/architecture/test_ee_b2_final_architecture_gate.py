# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — closure architecture gates (docs, modules, chaos SSOT)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.unit.runtime.architecture._ee_b2_final_facts import (
    EE_B2_BASE_TEST_MODULES,
    EE_B2_FINAL_QUALIFICATION,
    EE_B2_FINAL_TEST_MODULES,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_CHAOS_SUPPORT = _REPO / "testing_support" / "chaos"
_INTERGRAX = _REPO / "intergrax"
_EE_B2_CERT = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EE_B2_CHAOS_ENGINEERING_FAULT_INJECTION_CERTIFICATION.md"
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


def test_ee_b2_final_qualification_document_present() -> None:
    assert EE_B2_FINAL_QUALIFICATION.is_file()
    assert _EE_B2_CERT.is_file()


def test_ee_b2_final_all_gate_modules_present() -> None:
    for rel in EE_B2_BASE_TEST_MODULES + EE_B2_FINAL_TEST_MODULES:
        assert (_REPO / rel).is_file(), rel


def test_ee_b2_final_no_forbidden_chaos_runtime_in_production() -> None:
    hits: list[str] = []
    for path in _INTERGRAX.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for sym in _FORBIDDEN_PRODUCTION:
            if sym in text:
                hits.append(f"{path.relative_to(_REPO)}:{sym}")
    assert hits == []


def test_ee_b2_final_production_does_not_import_testing_support_chaos() -> None:
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


def test_ee_b2_final_chaos_helpers_no_identity_mint() -> None:
    mint_pattern = re.compile(
        r"\b(mint_run_id|mint_attempt_id|mint_execution_id|mint_task_id)\s*\(",
    )
    for path in _CHAOS_SUPPORT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert mint_pattern.search(text) is None, path.name
