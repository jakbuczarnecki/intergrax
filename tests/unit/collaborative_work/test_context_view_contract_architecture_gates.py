# © Artur Czarnecki. All rights reserved.

"""MP-5B — architecture gates for ContextView public contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTEXT_VIEW_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "context_view.py"

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
    "applications.",
    "intergrax.memory.",
    "intergrax.rag.",
    "intergrax.context.",
)
_FORBIDDEN_ANY_NAMES = frozenset({"Any"})


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_any_usage(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_ANY_NAMES:
            violations.append(f"{path.name}:{node.lineno} references Any")
    return violations


def test_context_view_contract_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_CONTEXT_VIEW_CONTRACT):
        if any(
            module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES
        ):
            violations.append(f"{_CONTEXT_VIEW_CONTRACT.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_context_view_contract_has_no_public_any() -> None:
    violations = _collect_any_usage(_CONTEXT_VIEW_CONTRACT)
    assert not violations, "\n".join(violations)


def test_no_context_view_storage_class_in_contract_module() -> None:
    text = _CONTEXT_VIEW_CONTRACT.read_text(encoding="utf-8")
    forbidden_names = (
        "ContextViewDatabase",
        "ContextViewMemoryStore",
        "ContextViewVectorDatabase",
        "ContextViewStorage",
    )
    for name in forbidden_names:
        assert name not in text, f"forbidden storage symbol present: {name}"
