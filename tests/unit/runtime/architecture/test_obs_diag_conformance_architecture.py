# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-CONFORMANCE — Evidence → Reconstruction → Diagnostics architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIAG_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_RECONSTRUCTION_PKG = _REPO_ROOT / "intergrax" / "runtime" / "observability" / "reconstruction"
_ORCHESTRATOR = _DIAG_ROOT / "diagnostic_orchestrator.py"

_EXECUTION_CANONICAL_SPINE = (
    "diagnostic_orchestrator.py",
    "lifecycle_analysis.py",
    "diagnostic_assessment.py",
    "execution_failure_analysis.py",
    "problem_grouping.py",
    "problem_lifecycle.py",
    "terminal_execution_diagnostic_trigger.py",
)

_FORBIDDEN_ORCHESTRATOR_IMPORT_PREFIXES = (
    "intergrax.runtime.events.persistence_contract",
    "intergrax.runtime.observability.evidence_persistence",
)

_FORBIDDEN_ORCHESTRATOR_SYMBOLS = frozenset(
    {
        "EvidencePersistencePort",
        "RuntimeEventPersistence",
        "RuntimeEventStore",
        "ExecutionLineagePersistence",
        "CausalEvidencePersistence",
        "FunctionalEvidencePersistence",
    }
)

_CONFORMANCE_HELPER_SUFFIXES = (
    "persistence_conformance.py",
    "functional_evidence_persistence_conformance.py",
)

_MINT_EXECUTION_PATTERN = re.compile(
    r"\b(mint_execution_id|mint_attempt_id|mint_run_id|mint_task_id)\s*\(",
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
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


def test_single_execution_reconstructor_definition_in_production() -> None:
    definitions: list[Path] = []
    for path in _python_files_under(_REPO_ROOT / "intergrax"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ExecutionReconstructor":
                definitions.append(path)
    assert definitions == [_RECONSTRUCTION_PKG / "execution_reconstruction.py"]


def test_reconstruction_package_does_not_import_diagnostics() -> None:
    for path in _python_files_under(_RECONSTRUCTION_PKG):
        text = path.read_text(encoding="utf-8")
        assert "intergrax.runtime.diagnostics" not in text, path.relative_to(_REPO_ROOT)


def test_diagnostic_orchestrator_imports_shared_reconstruction_only() -> None:
    imports = _module_imports(_ORCHESTRATOR)
    assert "intergrax.runtime.observability.reconstruction" in imports


def test_diagnostic_orchestrator_does_not_import_execution_persistence_ports() -> None:
    text = _ORCHESTRATOR.read_text(encoding="utf-8")
    for symbol in _FORBIDDEN_ORCHESTRATOR_SYMBOLS:
        assert symbol not in text, f"orchestrator must not reference {symbol}"
    for prefix in _FORBIDDEN_ORCHESTRATOR_IMPORT_PREFIXES:
        assert prefix not in text, f"orchestrator must not import {prefix}"


def test_execution_diagnostic_spine_does_not_reference_trace_event() -> None:
    for name in _EXECUTION_CANONICAL_SPINE:
        path = _DIAG_ROOT / name
        assert path.is_file(), name
        assert "TraceEvent" not in path.read_text(encoding="utf-8"), name


def test_lifecycle_analyzer_does_not_import_runtime_event_persistence() -> None:
    path = _DIAG_ROOT / "lifecycle_analysis.py"
    imports = _module_imports(path)
    assert "RuntimeEventPersistence" not in path.read_text(encoding="utf-8")
    assert not any(
        mod.endswith("persistence_contract") or mod.endswith("runtime_event_store")
        for mod in imports
    )


def test_diagnostics_spine_does_not_mint_execution_identity() -> None:
    violations: list[str] = []
    for path in _python_files_under(_DIAG_ROOT):
        if path.name in _CONFORMANCE_HELPER_SUFFIXES:
            continue
        if _MINT_EXECUTION_PATTERN.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(_REPO_ROOT)))
    assert violations == []


def test_diagnostic_orchestrator_signal_path_does_not_call_reconstructor() -> None:
    tree = ast.parse(_ORCHESTRATOR.read_text(encoding="utf-8-sig"))
    signal_method: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DiagnosticOrchestrator":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_analyze_signal_subject_scope":
                    signal_method = item
    assert signal_method is not None
    source = ast.get_source_segment(_ORCHESTRATOR.read_text(encoding="utf-8-sig"), signal_method)
    assert source is not None
    assert "reconstruct_execution" not in source
    assert "ExecutionReconstructor" not in source
