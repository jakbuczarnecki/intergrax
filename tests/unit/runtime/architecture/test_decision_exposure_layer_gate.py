# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_CONTRACT_FILES = (
    _REPO_ROOT / "intergrax" / "contracts" / "decision_authoritative_exposure.py",
    _REPO_ROOT / "intergrax" / "contracts" / "decision_exposure_selection.py",
)
_RUNTIME_FILES = (
    _REPO_ROOT / "intergrax" / "runtime" / "decision_exposure_mapping.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "host_terminal_decision_exposure_selector.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "decision_exposure_collector.py",
)


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_contracts_do_not_import_runtime() -> None:
    violations: list[str] = []
    for path in _CONTRACT_FILES:
        for module in _collect_imports(path):
            if module.startswith("intergrax.runtime"):
                violations.append(f"{path.name} imports {module}")
    assert not violations


def test_runtime_mapper_does_not_import_applications() -> None:
    mapper = _REPO_ROOT / "intergrax" / "runtime" / "decision_exposure_mapping.py"
    violations = [
        module
        for module in _collect_imports(mapper)
        if module.startswith("applications.") or module.startswith("platform_proofs.")
    ]
    assert not violations


def test_selector_does_not_import_scenario_modules() -> None:
    selector = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "execution"
        / "host_terminal_decision_exposure_selector.py"
    )
    violations = [
        module
        for module in _collect_imports(selector)
        if "scenario" in module.lower() or module.startswith("platform_proofs.")
    ]
    assert not violations
