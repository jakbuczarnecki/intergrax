# © Artur Czarnecki. All rights reserved.

"""MP-5C — architecture gates for ContextView visibility policy."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "context_view_visibility_policy.py"
)
_DEFAULT_POLICY = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "context_view_visibility.py"
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
    "applications.",
    "intergrax.memory.",
    "intergrax.rag.",
    "intergrax.context.",
)
_FORBIDDEN_ANY_NAMES = frozenset({"Any"})
_FORBIDDEN_DYNAMIC_NAMES = frozenset({"getattr", "setattr", "hasattr"})


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


def _collect_forbidden_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_ANY_NAMES:
            violations.append(f"{path.name}:{node.lineno} references Any")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_DYNAMIC_NAMES:
            violations.append(f"{path.name}:{node.lineno} references {node.id}")
    return violations


def _assert_no_forbidden_imports(path: Path) -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(path):
        if any(
            module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES
        ):
            violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp5c_policy_contract_has_no_forbidden_imports() -> None:
    _assert_no_forbidden_imports(_POLICY_CONTRACT)


def test_mp5c_default_policy_has_no_forbidden_imports() -> None:
    _assert_no_forbidden_imports(_DEFAULT_POLICY)


def test_mp5c_policy_contract_has_no_public_any_or_dynamic_tricks() -> None:
    violations = _collect_forbidden_names(_POLICY_CONTRACT)
    assert not violations, "\n".join(violations)


def test_mp5c_default_policy_has_no_public_any_or_dynamic_tricks() -> None:
    violations = _collect_forbidden_names(_DEFAULT_POLICY)
    assert not violations, "\n".join(violations)


def test_mp5c_public_policy_contract_symbols_exist() -> None:
    text = _POLICY_CONTRACT.read_text(encoding="utf-8")
    required = (
        "class ContextViewVisibilityPolicy",
        "class ContextViewPolicyDecision",
        "class ContextViewVisibilityPolicyInput",
    )
    for symbol in required:
        assert symbol in text, f"missing {symbol}"


def test_mp5c_closed_mp5d_next_docs_markers() -> None:
    docs = (
        _REPO_ROOT / "docs" / "project" / "architecture" / "COLLABORATIVE_WORK.md",
        _REPO_ROOT / "docs" / "project" / "maintainers" / "plans" / "COLLABORATIVE_WORK.md",
        _REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md",
        _REPO_ROOT / "docs" / "project" / "capabilities" / "plan" / "MULTIPLAYER_AI.md",
    )
    for path in docs:
        text = path.read_text(encoding="utf-8-sig")
        assert "MP-5C — APPROVED / CLOSED" in text, path.name
        assert "MP-5D — NEXT" in text, path.name
        assert "context_view_visibility_policy.py" in text, path.name
