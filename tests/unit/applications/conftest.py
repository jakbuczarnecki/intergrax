# © Artur Czarnecki. All rights reserved.

"""Autouse env for diagnostic Problem list cursor secret in application unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from testing_support.application_environment_test_support import stub_environment_llm_adapter
from testing_support.research_llm_test_support import configured_research_llm

_CURSOR_SECRET_ENV = "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"
_CURSOR_SECRET_VALUE = "unit-test-diagnostic-problem-list-cursor-secret"

__all__ = ["configured_research_llm"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def pytest_configure(_config: pytest.Config) -> None:
    """Register shipped product manifests before module-level parametrize collection."""
    for path in (_REPO_ROOT, _REPO_ROOT / "applications", _REPO_ROOT / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    import shipped_product_manifest_catalog  # noqa: F401


@pytest.fixture(autouse=True)
def _stub_tier3_environment_llm_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_environment_llm_adapter(monkeypatch)


@pytest.fixture(autouse=True)
def _diagnostic_problem_list_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_CURSOR_SECRET_ENV, _CURSOR_SECRET_VALUE)
