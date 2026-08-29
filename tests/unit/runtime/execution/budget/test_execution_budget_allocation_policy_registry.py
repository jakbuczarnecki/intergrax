# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — execution budget allocation policy resolver and entry-point registry tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
    load_execution_budget_allocation_policy,
    resolve_execution_budget_allocation_policy,
    resolve_execution_budget_allocation_policy_from_runtime_config,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig

pytestmark = pytest.mark.unit


class _EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        return self._value


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


class _InvalidBudgetPolicy:
    pass


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
    monkeypatch.setattr(
        "intergrax.runtime.execution.budget.registry.entry_points",
        lambda group=None: [_EntryPoint("custom", _CustomBudgetPolicy)],
    )
    resolved = resolve_execution_budget_allocation_policy(entry_point_policy_id="custom")
    assert isinstance(resolved, _CustomBudgetPolicy)


def test_load_execution_budget_allocation_policy_instantiates_class_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "intergrax.runtime.execution.budget.registry.entry_points",
        lambda group=None: [_EntryPoint("custom", _CustomBudgetPolicy)],
    )
    loaded = load_execution_budget_allocation_policy("custom")
    assert isinstance(loaded, _CustomBudgetPolicy)


def test_resolve_missing_entry_point_fails_closed() -> None:
    with pytest.raises(ExecutionBudgetAllocationPolicyConfigurationError, match="not found"):
        resolve_execution_budget_allocation_policy(entry_point_policy_id="missing-policy")


def test_load_invalid_entry_point_object_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "intergrax.runtime.execution.budget.registry.entry_points",
        lambda group=None: [_EntryPoint("bad", _InvalidBudgetPolicy)],
    )
    with pytest.raises(TypeError, match="ExecutionBudgetAllocationPolicy"):
        load_execution_budget_allocation_policy("bad")


def test_resolve_from_runtime_config_prefers_instance_override() -> None:
    custom = _CustomBudgetPolicy()
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_budget_allocation_policy = custom
    config.execution_budget_allocation_policy_id = "ignored"
    resolved = resolve_execution_budget_allocation_policy_from_runtime_config(config)
    assert resolved is custom


def test_entry_point_group_name() -> None:
    from intergrax.runtime.execution.budget import registry

    assert registry._ENTRY_POINT_GROUP == "intergrax.execution_budget_allocation_policies"
