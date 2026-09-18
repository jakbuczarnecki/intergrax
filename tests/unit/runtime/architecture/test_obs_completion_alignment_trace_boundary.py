# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-RECERT-P2A-R2 — completion alignment trace ownership architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.nexus.tracing.execution.completion_alignment import (
    CompletionAlignmentDiagV1,
)
from intergrax.runtime.observability.qualification_runtime_trace import (
    TaskTraceRuntimeDiagnosticPort,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OBSERVABILITY_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "observability"
_CANONICAL = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tracing"
    / "execution"
    / "completion_alignment.py"
)
_OLD_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "diagnostics"
    / "completion_alignment_diag.py"
)
_FORBIDDEN_DIAG_PREFIX = "intergrax.runtime.diagnostics"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


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


def test_observability_package_does_not_import_runtime_diagnostics() -> None:
    for path in _iter_python_files(_OBSERVABILITY_ROOT):
        for module in _import_modules_in_file(path):
            if module == _FORBIDDEN_DIAG_PREFIX or module.startswith(
                f"{_FORBIDDEN_DIAG_PREFIX}."
            ):
                raise AssertionError(
                    f"{path.relative_to(_REPO_ROOT)} imports forbidden module {module}",
                )


def test_old_completion_alignment_diag_module_absent() -> None:
    assert not _OLD_MODULE.is_file()


def test_single_canonical_completion_alignment_diag_definition() -> None:
    definitions: list[Path] = []
    intergrax_root = _REPO_ROOT / "intergrax"
    for path in _iter_python_files(intergrax_root):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "CompletionAlignmentDiagV1"
            ):
                definitions.append(path)
    assert definitions == [_CANONICAL]


def test_completion_alignment_diag_canonical_module() -> None:
    assert CompletionAlignmentDiagV1.__module__ == (
        "intergrax.runtime.nexus.tracing.execution.completion_alignment"
    )


def test_runtime_diagnostic_trace_port_emit_completion_alignment_typed() -> None:
    from typing import get_type_hints

    hints = get_type_hints(
        TaskTraceRuntimeDiagnosticPort.emit_completion_alignment,
        globalns=globals(),
    )
    assert hints["payload"] is CompletionAlignmentDiagV1
