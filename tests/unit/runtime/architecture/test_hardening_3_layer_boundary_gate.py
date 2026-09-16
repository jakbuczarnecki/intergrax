# © Artur Czarnecki. All rights reserved.

"""HARDENING-3 — contracts must not depend on runtime (one-way layer boundary)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax" / "contracts"

# Documented deferred migrations — see LAYER_BOUNDARY_HARDENING.md
_ALLOWLISTED_RUNTIME_IMPORTS: frozenset[str] = frozenset(
    {
        "intergrax/contracts/execution_evidence/persistence_port.py",
        "intergrax/contracts/host_profile_slices.py",
        "intergrax/contracts/runtime_cost.py",
        "intergrax/contracts/runtime_execution_context.py",
        "intergrax/contracts/runtime_mapping.py",
    }
)


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


def test_hardening_3_contracts_do_not_import_runtime_except_allowlist() -> None:
    violations: list[str] = []
    for path in sorted(_CONTRACTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        runtime_imports = _collect_runtime_imports(path)
        if not runtime_imports:
            continue
        if rel in _ALLOWLISTED_RUNTIME_IMPORTS:
            continue
        for module in runtime_imports:
            violations.append(f"{rel}: {module}")
    assert violations == [], "contracts → runtime coupling:\n" + "\n".join(violations)


def test_hardening_3_self_healing_contracts_have_no_runtime_imports() -> None:
    root = _CONTRACTS_ROOT / "self_healing"
    if not root.is_dir():
        return
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_runtime_imports(path):
            violations.append(f"{rel}: {module}")
    assert violations == []
