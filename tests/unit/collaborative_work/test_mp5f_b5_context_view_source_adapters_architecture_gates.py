# © Artur Czarnecki. All rights reserved.

"""MP-5F-B5 — architecture gates for ContextView source adapters."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAPTER = _REPO_ROOT / "intergrax" / "collaborative_work" / "context_view_source_adapters.py"
_MAPPING = _REPO_ROOT / "intergrax" / "collaborative_work" / "context_view_source_mapping.py"
_COMPOSER = _REPO_ROOT / "intergrax" / "collaborative_work" / "context_view_composition.py"

_FORBIDDEN_ADAPTER_IMPORT_PREFIXES = (
    "intergrax.memory.default_",
    "intergrax.memory.stores",
    "intergrax.memory.sqlite",
    "intergrax.rag.vectorstore.providers",
    "intergrax.collaborative_work.sqlite_repository",
    "intergrax.collaborative_work.postgresql_repository",
    "intergrax.collaborative_work.in_memory_repository",
    "intergrax.runtime.nexus",
    "applications.",
)


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


def test_mp5f_b5_adapters_have_no_concrete_repo_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_ADAPTER):
        if any(module.startswith(prefix) for prefix in _FORBIDDEN_ADAPTER_IMPORT_PREFIXES):
            violations.append(f"{_ADAPTER.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp5f_b5_mapping_has_no_concrete_repo_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_MAPPING):
        if any(module.startswith(prefix) for prefix in _FORBIDDEN_ADAPTER_IMPORT_PREFIXES):
            violations.append(f"{_MAPPING.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp5f_b5_composer_does_not_import_default_adapters() -> None:
    text = _COMPOSER.read_text(encoding="utf-8")
    for symbol in (
        "DefaultMemoryContextSource",
        "context_view_source_adapters",
        "context_view_source_wiring",
    ):
        assert symbol not in text, symbol
