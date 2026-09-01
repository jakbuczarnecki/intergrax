# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — execution budget allocation policy resolver and entry-point registry tests."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.core.plugins.discovery import (
    EP_EXECUTION_BUDGET_ALLOCATION_POLICIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.runtime.execution.budget.models import (
    ChildBudgetAllocationContext,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
)
from intergrax.runtime.execution.budget.policy import (
    DefaultSharedPoolBudgetPolicy,
    ExecutionBudgetAllocationPolicy,
)
from intergrax.runtime.execution.budget.registry import (
    ExecutionBudgetAllocationPolicyConfigurationError,
    list_execution_budget_allocation_policy_ids,
    load_execution_budget_allocation_policy,
    resolve_execution_budget_allocation_policy,
    resolve_execution_budget_allocation_policy_from_runtime_config,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig

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


class _CustomBudgetPolicy:
    def resolve_child_budget(
        self,
        context: ChildBudgetAllocationContext,
    ) -> ChildBudgetAllocationDecision:
        _ = context
        return ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.RESERVED,
            reservation_request=RunBudget(max_tool_calls=1),
        )


_CUSTOM_BUDGET_POLICY_INSTANCE = _CustomBudgetPolicy()


class _InvalidBudgetPolicy:
    pass


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _policy_ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", EP_EXECUTION_BUDGET_ALLOCATION_POLICIES)


def test_resolve_prefers_explicit_instance() -> None:
    custom = _CustomBudgetPolicy()
    resolved = resolve_execution_budget_allocation_policy(
        policy_override=custom,
        entry_point_policy_id="would-fail-if-used",
    )
    assert resolved is custom


def test_resolve_without_configuration_returns_default() -> None:
    resolved = resolve_execution_budget_allocation_policy()
    assert isinstance(resolved, DefaultSharedPoolBudgetPolicy)


def test_resolve_loads_valid_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_policy_ep("custom", "_CustomBudgetPolicy")])
    resolved = resolve_execution_budget_allocation_policy(entry_point_policy_id="custom")
    assert isinstance(resolved, _CustomBudgetPolicy)


def test_load_execution_budget_allocation_policy_instantiates_class_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_policy_ep("custom", "_CustomBudgetPolicy")])
    loaded = load_execution_budget_allocation_policy("custom")
    assert isinstance(loaded, _CustomBudgetPolicy)


def test_load_instance_entry_point_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_policy_ep("custom", "_CUSTOM_BUDGET_POLICY_INSTANCE")])
    loaded = load_execution_budget_allocation_policy("custom")
    assert loaded is _CUSTOM_BUDGET_POLICY_INSTANCE


def test_load_missing_entry_point_returns_none() -> None:
    assert load_execution_budget_allocation_policy("missing-policy") is None


def test_resolve_missing_entry_point_fails_closed() -> None:
    with pytest.raises(ExecutionBudgetAllocationPolicyConfigurationError, match="not found"):
        resolve_execution_budget_allocation_policy(entry_point_policy_id="missing-policy")


def test_load_invalid_entry_point_object_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_policy_ep("bad", "_InvalidBudgetPolicy")])
    with pytest.raises(TypeError, match="ExecutionBudgetAllocationPolicy"):
        load_execution_budget_allocation_policy("bad")


def test_list_execution_budget_allocation_policy_ids_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _policy_ep("beta", "_CustomBudgetPolicy"),
            _policy_ep("alpha", "_CustomBudgetPolicy"),
        ],
    )
    assert list_execution_budget_allocation_policy_ids() == ("alpha", "beta")


def test_resolve_from_runtime_config_prefers_instance_override() -> None:
    custom = _CustomBudgetPolicy()
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_budget_allocation_policy = custom
    config.execution_budget_allocation_policy_id = "ignored"
    resolved = resolve_execution_budget_allocation_policy_from_runtime_config(config)
    assert resolved is custom


def test_entry_point_group_name() -> None:
    from intergrax.core.plugins.discovery import EP_EXECUTION_BUDGET_ALLOCATION_POLICIES
    from intergrax.runtime.execution.budget import registry

    assert registry._ENTRY_POINT_GROUP == EP_EXECUTION_BUDGET_ALLOCATION_POLICIES
