# © Artur Czarnecki. All rights reserved.

"""OBS-CONTRACT-BOUNDARY-1 — execution reconstruction contract ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.runtime.observability.reconstruction import (
    ExecutionReconstruction as LegacyExecutionReconstruction,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACTS_RECONSTRUCTION = _REPO_ROOT / "intergrax" / "contracts"
_DIAG_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"

_FORBIDDEN_DIAG_RECONSTRUCTION_PREFIX = "intergrax.runtime.observability.reconstruction"


def _collect_runtime_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime"):
                hits.append(node.module)
    return hits


def test_execution_reconstruction_contract_modules_do_not_import_runtime() -> None:
    violations: list[str] = []
    patterns = (
        "execution_reconstruction.py",
        "execution_reconstruction_models.py",
        "execution_reconstruction_lineage.py",
    )
    for name in patterns:
        path = _CONTRACTS_RECONSTRUCTION / name
        for module in _collect_runtime_imports(path):
            violations.append(f"{name}: {module}")
    assert violations == []


def test_legacy_runtime_reconstruction_result_is_canonical_contract_type() -> None:
    assert LegacyExecutionReconstruction is ExecutionReconstruction


def test_diagnostics_core_does_not_import_runtime_reconstruction_dto() -> None:
    violations: list[str] = []
    skip_names = {
        "persistence_conformance.py",
        "functional_evidence_persistence_conformance.py",
        "__init__.py",
    }
    for path in sorted(_DIAG_ROOT.rglob("*.py")):
        if path.name in skip_names or "__pycache__" in path.parts:
            continue
        if path.parent.name == "providers":
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from ") and not stripped.startswith("import "):
                continue
            if _FORBIDDEN_DIAG_RECONSTRUCTION_PREFIX in stripped:
                if "ExecutionReconstructor" in stripped:
                    continue
                violations.append(f"{path.relative_to(_REPO_ROOT)}: {stripped}")
    assert violations == []
