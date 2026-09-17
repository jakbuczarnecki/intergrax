# © Artur Czarnecki. All rights reserved.

"""MP-5D — architecture gates for ContextView source composition ports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PORTS_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "context_view_source_ports.py"
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
_FORBIDDEN_IMPLEMENTATION_NAMES = (
    "DefaultMemoryContextSource",
    "DefaultKnowledgeContextSource",
    "DefaultUclContextSource",
    "DefaultCollaborativeWorkContextSource",
    "ContextViewComposer",
)
_FORBIDDEN_RETRIEVAL_SYMBOLS = (
    "Repository",
    "retrieve_candidates",
    "hydrate",
    "MemoryRecord",
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


def _collect_forbidden_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_ANY_NAMES:
            violations.append(f"{path.name}:{node.lineno} references Any")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_DYNAMIC_NAMES:
            violations.append(f"{path.name}:{node.lineno} references {node.id}")
    return violations


def test_mp5d_source_ports_contract_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_SOURCE_PORTS_CONTRACT):
        if any(
            module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORT_PREFIXES
        ):
            violations.append(f"{_SOURCE_PORTS_CONTRACT.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp5d_source_ports_contract_has_no_any_or_dynamic_tricks() -> None:
    violations = _collect_forbidden_names(_SOURCE_PORTS_CONTRACT)
    assert not violations, "\n".join(violations)


def test_mp5d_no_default_adapters_or_composer_in_contract_module() -> None:
    text = _SOURCE_PORTS_CONTRACT.read_text(encoding="utf-8")
    for name in _FORBIDDEN_IMPLEMENTATION_NAMES:
        assert name not in text, f"forbidden implementation symbol: {name}"


def test_mp5d_no_retrieval_or_storage_symbols_in_contract_module() -> None:
    text = _SOURCE_PORTS_CONTRACT.read_text(encoding="utf-8")
    for symbol in _FORBIDDEN_RETRIEVAL_SYMBOLS:
        assert symbol not in text, f"forbidden retrieval/storage symbol: {symbol}"


def test_mp5d_separate_domain_source_ports_exist() -> None:
    text = _SOURCE_PORTS_CONTRACT.read_text(encoding="utf-8")
    required = (
        "class MemoryContextSourcePort",
        "class KnowledgeContextSourcePort",
        "class UclContextSourcePort",
        "class CollaborativeWorkContextSourcePort",
        "class ContextViewMemorySourceCandidate",
        "class ContextViewKnowledgeSourceCandidate",
    )
    for symbol in required:
        assert symbol in text, f"missing {symbol}"
