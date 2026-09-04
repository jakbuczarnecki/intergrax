# © Artur Czarnecki. All rights reserved.

"""Integration runtime test fixtures."""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.integration.runtime.diag_final_otel_support"]

_CURSOR_SECRET_ENV = "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"
_CURSOR_SECRET_VALUE = "integration-test-diagnostic-problem-list-cursor-secret"


@pytest.fixture(autouse=True)
def _diagnostic_problem_list_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_CURSOR_SECRET_ENV, _CURSOR_SECRET_VALUE)
