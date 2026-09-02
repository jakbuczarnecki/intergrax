# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for generic source selection modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_GENERIC_DIR = _REPO_ROOT / "agents" / "web_search_qualifier" / "source_selection"
_GENERIC_MODULES = (
    "contracts.py",
    "engine.py",
    "llm_selector.py",
    "matching.py",
    "url_normalization.py",
)
_FORBIDDEN_LITERALS = (
    "python.org",
    "python-3120",
    "/downloads/release/",
    "rc3",
    "Q3",
    "qualification",
    "functional_diagnostics_q3",
)
_FORBIDDEN_IMPORT_PREFIXES = (
    "functional_diagnostics_q3",
    "web_search_qualifier.url_identity",
)


def _module_text(name: str) -> str:
    return (_GENERIC_DIR / name).read_text(encoding="utf-8")


def _module_imports(name: str) -> set[str]:
    tree = ast.parse(_module_text(name))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


@pytest.mark.parametrize("module_name", _GENERIC_MODULES)
def test_generic_modules_have_no_domain_literals(module_name: str) -> None:
    lowered = _module_text(module_name).lower()
    for literal in _FORBIDDEN_LITERALS:
        assert literal.lower() not in lowered, f"{module_name} contains {literal}"


@pytest.mark.parametrize("module_name", _GENERIC_MODULES)
def test_generic_modules_have_no_qualification_imports(module_name: str) -> None:
    imports = _module_imports(module_name)
    for module in imports:
        lowered = module.lower()
        for forbidden in _FORBIDDEN_IMPORT_PREFIXES:
            assert forbidden not in lowered, f"{module_name} imports {module}"


def test_extension_policy_composes_without_engine_changes() -> None:
    from web_search_qualifier.source_selection.engine import SourceSelectionEngine
    from web_search_qualifier.source_selection.example_docs_policy import ExampleDocsSourceSelectionPolicy
    from web_search_qualifier.web_search import WebSearchCandidate

    engine = SourceSelectionEngine(
        policies=(ExampleDocsSourceSelectionPolicy(),),
        llm_selector=None,
    )
    decision = engine.select(
        run_id="run-ext",
        task_message="docs",
        candidates=(
            WebSearchCandidate(
                rank=1,
                url="https://vendor.example.net/",
                title="vendor",
                snippet="",
                provider="test",
            ),
            WebSearchCandidate(
                rank=2,
                url="https://docs.example.com/guide",
                title="guide",
                snippet="",
                provider="test",
            ),
        ),
    )
    assert decision.selected_url == "https://docs.example.com/guide"
