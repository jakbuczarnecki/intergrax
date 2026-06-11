# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

_RESEARCH_HARNESS_API_KEY = "gate-test-harness-key"


@pytest.fixture(autouse=True)
def _research_harness_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", _RESEARCH_HARNESS_API_KEY)
