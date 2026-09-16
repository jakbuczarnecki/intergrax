# © Artur Czarnecki. All rights reserved.

"""OBS-CONTRACT-BOUNDARY-2 — platform problem signal contract ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.platform_problem_signal import PlatformProblemSignal
from intergrax.contracts.functional_validation_evidence import (
    FunctionalValidationEvidence,
)
from intergrax.runtime.observability.functional_validation_evidence import (
    FunctionalValidationEvidence as LegacyFunctionalValidationEvidence,
)
from intergrax.runtime.observability.problem_signal import (
    PlatformProblemSignal as LegacyPlatformProblemSignal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_CLUSTER_ROOTS = (
    _REPO_ROOT / "intergrax" / "contracts" / "platform_problem_signal.py",
    _REPO_ROOT / "intergrax" / "contracts" / "functional_validation_evidence.py",
    _REPO_ROOT / "intergrax" / "contracts" / "application_observability_attributes.py",
)
_DIAG_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_FORBIDDEN_DIAG_SIGNAL_IMPORT = "intergrax.runtime.observability.problem_signal"
_FORBIDDEN_DIAG_FVE_IMPORT = (
    "intergrax.runtime.observability.functional_validation_evidence"
)


def _import_modules_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def test_legacy_problem_signal_import_path_is_canonical_contract_type() -> None:
    assert LegacyPlatformProblemSignal is PlatformProblemSignal


def test_legacy_functional_validation_evidence_import_path_is_canonical() -> None:
    assert LegacyFunctionalValidationEvidence is FunctionalValidationEvidence


def test_canonical_platform_problem_signal_module_has_no_runtime_imports() -> None:
    forbidden_prefix = "intergrax.runtime"
    for path in _CONTRACT_CLUSTER_ROOTS:
        for module in _import_modules_in_file(path):
            assert not (
                module == forbidden_prefix or module.startswith(f"{forbidden_prefix}.")
            ), f"{path.relative_to(_REPO_ROOT)} imports forbidden module {module}"


def test_diagnostics_core_does_not_import_runtime_problem_signal_dto() -> None:
    violations: list[str] = []
    for path in _iter_python_files(_DIAG_ROOT):
        for module in _import_modules_in_file(path):
            if module == _FORBIDDEN_DIAG_SIGNAL_IMPORT or module.startswith(
                f"{_FORBIDDEN_DIAG_SIGNAL_IMPORT}.",
            ):
                violations.append(f"{path.relative_to(_REPO_ROOT)} -> {module}")
            if module == _FORBIDDEN_DIAG_FVE_IMPORT or module.startswith(
                f"{_FORBIDDEN_DIAG_FVE_IMPORT}.",
            ):
                violations.append(f"{path.relative_to(_REPO_ROOT)} -> {module}")
    assert violations == []


def test_single_canonical_platform_problem_signal_definition() -> None:
    assert (
        PlatformProblemSignal.__module__
        == "intergrax.contracts.platform_problem_signal"
    )


def test_single_canonical_functional_validation_evidence_definition() -> None:
    assert (
        FunctionalValidationEvidence.__module__
        == "intergrax.contracts.functional_validation_evidence"
    )
