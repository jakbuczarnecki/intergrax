# © Artur Czarnecki. All rights reserved.

"""Autouse env for diagnostic Problem list cursor secret in application unit tests."""

from __future__ import annotations

import pytest

from testing_support.research_llm_test_support import configured_research_llm

_CURSOR_SECRET_ENV = "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"
_CURSOR_SECRET_VALUE = "unit-test-diagnostic-problem-list-cursor-secret"

__all__ = ["configured_research_llm"]


@pytest.fixture(autouse=True)
def _diagnostic_problem_list_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_CURSOR_SECRET_ENV, _CURSOR_SECRET_VALUE)
