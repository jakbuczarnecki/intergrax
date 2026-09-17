# © Artur Czarnecki. All rights reserved.

"""MP-5E — architecture gates for default ContextView composer."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSITION_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "context_view_composition.py"
)
_DEFAULT_COMPOSER = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "context_view_composition.py"
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
_FORBIDDEN_ADAPTER_NAMES = (
    "DefaultMemoryContextSource",
    "DefaultKnowledgeContextSource",
    "DefaultUclContextSource",
    "DefaultCollaborativeWorkContextSource",
)
_FORBIDDEN_RETRIEVAL_SYMBOLS = ("hydrate", "retrieve_candidates", "MemoryRecord")


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


def _collect_concrete_source_construction(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_ADAPTER_NAMES:
                violations.append(f"{path.name}:{node.lineno} constructs {node.func.id}")
    return violations


def _assert_no_forbidden_imports(path: Path) -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(path):
        if any(
            module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES
        ):
            violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("path", [_COMPOSITION_CONTRACT, _DEFAULT_COMPOSER])
def test_mp5e_modules_have_no_forbidden_imports(path: Path) -> None:
    _assert_no_forbidden_imports(path)


@pytest.mark.parametrize("path", [_COMPOSITION_CONTRACT, _DEFAULT_COMPOSER])
def test_mp5e_modules_have_no_any_or_dynamic_tricks(path: Path) -> None:
    violations = _collect_forbidden_names(path)
    assert not violations, "\n".join(violations)


def test_mp5e_public_composer_contract_exists() -> None:
    text = _COMPOSITION_CONTRACT.read_text(encoding="utf-8")
    assert "class ContextViewComposer" in text
    assert "class ContextViewCompositionRequest" in text


def test_mp5e_default_composer_does_not_construct_concrete_sources() -> None:
    violations = _collect_concrete_source_construction(_DEFAULT_COMPOSER)
    assert not violations, "\n".join(violations)


def test_mp5e_no_retrieval_symbols_in_composer_modules() -> None:
    for path in (_COMPOSITION_CONTRACT, _DEFAULT_COMPOSER):
        text = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_RETRIEVAL_SYMBOLS:
            assert symbol not in text, f"{path.name} references {symbol}"
