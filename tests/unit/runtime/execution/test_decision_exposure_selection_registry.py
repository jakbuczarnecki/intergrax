# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.discovery import (
    EP_DECISION_EXPOSURE_SELECTION_STRATEGIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.runtime.execution.decision_exposure_selection_registry import (
    list_decision_exposure_selection_strategy_ids,
)

pytestmark = pytest.mark.unit


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def test_list_strategy_ids_is_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _EntryPoint("listed.id", "some.module:Plugin", EP_DECISION_EXPOSURE_SELECTION_STRATEGIES)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([ep]))
    assert list_decision_exposure_selection_strategy_ids() == ("listed.id",)
