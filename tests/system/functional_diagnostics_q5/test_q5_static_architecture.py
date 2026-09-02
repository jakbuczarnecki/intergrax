# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for DIAG-FUNCTIONAL-Q5."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_QUALIFICATION_DIR = _REPO_ROOT / "intergrax" / "core" / "qualification"
_ANALYZER_PATH = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "functional_diagnostic_analyzer.py"
_FORBIDDEN_DOMAIN_IMPORT_PREFIXES = (
    "functional_diagnostics_q1",
    "functional_diagnostics_q2",
    "functional_diagnostics_q3",
    "functional_diagnostics_q4",
    "qdrant",
    "tavily",
    "web_search_qualifier",
    "tool_selection_qualifier",
    "model_routing_qualifier",
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_core_qualification_has_no_domain_imports() -> None:
    for path in _CORE_QUALIFICATION_DIR.glob("functional_qualification_*.py"):
        imports = _module_imports(path)
        for module in imports:
            lowered = module.lower()
            for forbidden in _FORBIDDEN_DOMAIN_IMPORT_PREFIXES:
                assert forbidden not in lowered, f"{path.name} imports {module}"


def test_analyzer_has_no_domain_imports() -> None:
    imports = _module_imports(_ANALYZER_PATH)
    for module in imports:
        lowered = module.lower()
        for forbidden in _FORBIDDEN_DOMAIN_IMPORT_PREFIXES:
            assert forbidden not in lowered, f"analyzer imports {module}"


_SOURCE_SELECTION_GENERIC_DIR = (
    _REPO_ROOT / "agents" / "web_search_qualifier" / "source_selection"
)
_SOURCE_SELECTION_GENERIC_MODULES = (
    "contracts.py",
    "engine.py",
    "llm_selector.py",
    "matching.py",
    "url_normalization.py",
)


def test_source_selection_engine_has_no_domain_literals() -> None:
    forbidden_literals = (
        "python.org",
        "python-3120",
        "/downloads/release/",
        "functional_diagnostics_q3",
    )
    for module_name in _SOURCE_SELECTION_GENERIC_MODULES:
        text = (_SOURCE_SELECTION_GENERIC_DIR / module_name).read_text(encoding="utf-8").lower()
        for literal in forbidden_literals:
            assert literal not in text, f"{module_name} contains {literal}"


def test_check_ids_are_namespaced_across_domains() -> None:
    from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
        CHECK_C1_CANDIDATES,
    )
    from intergrax.runtime.diagnostics.specifications.q2_tool_selection_functional_diagnostic_specification import (
        CHECK_Q2_CANDIDATES,
    )
    from intergrax.runtime.diagnostics.specifications.q3_web_search_functional_diagnostic_specification import (
        CHECK_Q3_CANDIDATES,
    )
    from intergrax.runtime.diagnostics.specifications.q4_model_routing_functional_diagnostic_specification import (
        CHECK_Q4_CANDIDATES,
    )
    ids = {CHECK_C1_CANDIDATES, CHECK_Q2_CANDIDATES, CHECK_Q3_CANDIDATES, CHECK_Q4_CANDIDATES}
    assert len(ids) == 4
