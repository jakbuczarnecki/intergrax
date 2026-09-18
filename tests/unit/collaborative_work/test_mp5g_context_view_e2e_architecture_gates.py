# © Artur Czarnecki. All rights reserved.

"""MP-5G — architecture gates for E2E qualification harness boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_HARNESS = _REPO / "tests" / "unit" / "collaborative_work" / "mp5g_e2e_harness.py"
_QUALIFICATION = (
    _REPO / "tests" / "unit" / "collaborative_work" / "test_mp5g_context_view_e2e_qualification.py"
)
_COMPOSER = _REPO / "intergrax" / "collaborative_work" / "context_view_composition.py"
_ADAPTER = _REPO / "intergrax" / "collaborative_work" / "context_view_source_adapters.py"

_FORBIDDEN_HARNESS_PREFIXES = (
    "intergrax.runtime.nexus",
    "applications.",
    "intergrax.collaborative_work.sqlite_repository",
)


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_mp5g_harness_has_no_nexus_or_application_imports() -> None:
    violations = [
        module
        for module in _collect_imports(_HARNESS)
        if any(module.startswith(prefix) for prefix in _FORBIDDEN_HARNESS_PREFIXES)
    ]
    assert not violations


def test_mp5g_qualification_tests_do_not_import_private_composer_internals() -> None:
    text = _QUALIFICATION.read_text(encoding="utf-8")
    assert "_collect_validated_candidates" not in text
    assert "monkeypatch" not in text


def test_mp5g_composer_still_has_no_default_adapter_imports() -> None:
    text = _COMPOSER.read_text(encoding="utf-8")
    for symbol in (
        "DefaultMemoryContextSource",
        "context_view_source_adapters",
        "context_view_source_wiring",
    ):
        assert symbol not in text


def test_mp5g_adapters_have_no_asyncio_run() -> None:
    text = _ADAPTER.read_text(encoding="utf-8")
    assert "asyncio.run" not in text
