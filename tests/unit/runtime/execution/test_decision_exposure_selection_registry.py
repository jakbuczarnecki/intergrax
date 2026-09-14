# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionDecision,
    DecisionExposureSelectionStrategy,
)
from intergrax.contracts.decision_authoritative_exposure import DecisionEvaluationScope
from intergrax.core.plugins.discovery import (
    EP_DECISION_EXPOSURE_SELECTION_STRATEGIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.runtime.execution.decision_exposure_selection_registry import (
    load_decision_exposure_selection_strategy,
)
from intergrax.runtime.execution.host_terminal_decision_exposure_selector import (
    HostTerminalDecisionExposureSelector,
)

pytestmark = pytest.mark.unit


class _CustomStrategy:
    @property
    def strategy_id(self) -> str:
        return "test.custom"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> DecisionExposureSelectionDecision[object]:
        inner = HostTerminalDecisionExposureSelector()
        outcome = inner.select(policy, candidates)
        if type(outcome).__name__ == "DecisionExposureSelectionFailure":
            return outcome
        return outcome


_CUSTOM_STRATEGY = _CustomStrategy()


class _NotAStrategy:
    pass


_NOT_A_STRATEGY = _NotAStrategy()


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


def test_custom_strategy_loads_via_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _EntryPoint(
        "test.custom",
        f"{__name__}:_CUSTOM_STRATEGY",
        EP_DECISION_EXPOSURE_SELECTION_STRATEGIES,
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([ep]))
    loaded = load_decision_exposure_selection_strategy("test.custom")
    assert loaded is not None
    assert isinstance(loaded, DecisionExposureSelectionStrategy)
    assert loaded.strategy_id == "test.custom"


def test_invalid_plugin_target_raises_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _EntryPoint("bad", f"{__name__}:_NOT_A_STRATEGY", EP_DECISION_EXPOSURE_SELECTION_STRATEGIES)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([ep]))
    with pytest.raises(TypeError):
        load_decision_exposure_selection_strategy("bad")
