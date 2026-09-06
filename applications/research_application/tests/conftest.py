# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

_RESEARCH_HARNESS_API_KEY = "gate-test-harness-key"
_DIAGNOSTIC_CURSOR_SECRET = "unit-test-diagnostic-problem-list-cursor-secret"


@pytest.fixture(autouse=True)
def _research_harness_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", _RESEARCH_HARNESS_API_KEY)
    monkeypatch.setenv(
        "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET",
        _DIAGNOSTIC_CURSOR_SECRET,
    )
