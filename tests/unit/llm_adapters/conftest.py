# © Artur Czarnecki. All rights reserved.

"""LLM adapter tests are part of the deterministic regression gate."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "/llm_adapters/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.gate)


pytestmark = pytest.mark.unit
