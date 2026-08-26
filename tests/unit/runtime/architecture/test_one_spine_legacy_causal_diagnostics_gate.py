# © Artur Czarnecki. All rights reserved.

"""Architecture gate — legacy synthetic causal diagnostics must not recur (ONE-SPINE-1)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "CausalDiagnosticChain",
        "CausalDiagnosticLink",
        "build_causal_diagnostic_chain",
        "resolve_causal_diagnostics_wiring",
        "CausalDiagnosticsWiring",
    }
)

_SCAN_ROOTS = (
  "intergrax/applications",
  "intergrax/runtime/observability",
  "scripts",
)

_EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        "tests",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _production_python_files() -> list[Path]:
    root = _repo_root()
    files: list[Path] = []
    for scan_root in _SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            files.append(path)
    return files


def _collect_forbidden_symbol_references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
            violations.append(f"{path.relative_to(_repo_root())}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            violations.append(
                f"{path.relative_to(_repo_root())}:{node.lineno} references .{node.attr}",
            )
    return violations


def _collect_trace_scope_causal_chain_violations(path: Path) -> list[str]:
    """Applications must not build synthetic causal diagnostic models from TraceScopeState."""
    if "intergrax/applications" not in path.as_posix():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "build_causal_diagnostic_chain":
                violations.append(
                    f"{path.relative_to(_repo_root())}:{node.lineno} calls build_causal_diagnostic_chain()",
                )
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("causal_diagnostics"):
                violations.append(
                    f"{path.relative_to(_repo_root())}:{node.lineno} imports from causal_diagnostics",
                )
    return violations


def test_no_legacy_causal_diagnostic_symbols_in_production_code() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        violations.extend(_collect_forbidden_symbol_references(path))
        violations.extend(_collect_trace_scope_causal_chain_violations(path))
    assert violations == []


def test_causal_diagnostics_module_removed() -> None:
    root = _repo_root()
    assert not (root / "intergrax/runtime/observability/causal_diagnostics.py").exists()
    assert not (root / "intergrax/applications/_shared/causal_diagnostics_wiring.py").exists()
