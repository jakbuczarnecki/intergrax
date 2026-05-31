# © Artur Czarnecki. All rights reserved.

"""LLM adapter tests are part of the deterministic regression gate."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if "/llm_adapters/" not in nodeid:
            continue
        if item.get_closest_marker("network"):
            continue
        item.add_marker(pytest.mark.gate)


pytestmark = pytest.mark.unit
