# © Artur Czarnecki. All rights reserved.

"""OBS-RECONSTRUCTION-1 — factual reconstruction ownership architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECONSTRUCTION_PKG = _REPO_ROOT / "intergrax" / "runtime" / "observability" / "reconstruction"
_OLD_EXECUTION = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "execution_reconstruction.py"
_OLD_LINEAGE = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "execution_lineage_reconstruction.py"
_HISTORICAL = _REPO_ROOT / "intergrax" / "runtime" / "observability" / "historical_reconstruction.py"
_DIAG_ORCHESTRATOR = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "diagnostic_orchestrator.py"


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _python_files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_reconstruction_package_has_no_diagnostics_imports() -> None:
    for path in _python_files_under(_RECONSTRUCTION_PKG):
        text = path.read_text(encoding="utf-8")
        assert "intergrax.runtime.diagnostics" not in text, path.relative_to(_REPO_ROOT)


def test_old_diagnostic_reconstruction_modules_absent() -> None:
    assert not _OLD_EXECUTION.is_file()
    assert not _OLD_LINEAGE.is_file()


def test_single_execution_reconstructor_definition() -> None:
    definitions: list[Path] = []
    for path in _python_files_under(_REPO_ROOT / "intergrax"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ExecutionReconstructor":
                definitions.append(path)
    assert definitions == [_RECONSTRUCTION_PKG / "execution_reconstruction.py"]


def test_historical_reconstruction_service_uses_neutral_reconstruction() -> None:
    imports = _module_imports(_HISTORICAL)
    assert "intergrax.runtime.observability.reconstruction" in imports
    assert "intergrax.runtime.diagnostics.execution_reconstruction" not in imports


def test_diagnostic_orchestrator_consumes_neutral_reconstruction() -> None:
    imports = _module_imports(_DIAG_ORCHESTRATOR)
    assert "intergrax.contracts.execution_reconstruction" in imports
    text = _DIAG_ORCHESTRATOR.read_text(encoding="utf-8")
    assert "ExecutionReconstructionReader" in text
    assert "ExecutionReconstructor" not in text
