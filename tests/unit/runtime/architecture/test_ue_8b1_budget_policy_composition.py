# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — RuntimeConfig budget allocation policy reaches GraphExecutor child runner."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.core.plugins.discovery import (
    EP_EXECUTION_BUDGET_ALLOCATION_POLICIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.runtime.execution.budget.models import (
    ChildBudgetAllocationContext,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
)
from intergrax.runtime.execution.budget.policy import DefaultSharedPoolBudgetPolicy
from intergrax.runtime.execution.budget.registry import (
    ExecutionBudgetAllocationPolicyConfigurationError,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.registry.agent_registry import AgentRegistry

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


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _policy_ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", EP_EXECUTION_BUDGET_ALLOCATION_POLICIES)


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


def _minimal_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults()


def _child_runner(loop: object) -> object:
    return loop._graph_executor._child_runner  # noqa: SLF001


def test_runtime_config_direct_instance_reaches_graph_executor_child_runner() -> None:
    custom = _CustomBudgetPolicy()
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_budget_allocation_policy = custom

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert _child_runner(loop)._budget_policy is custom


def test_runtime_config_entry_point_id_reaches_graph_executor_child_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_policy_ep("custom", "_CustomBudgetPolicy")])
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_budget_allocation_policy_id = "custom"

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert isinstance(_child_runner(loop)._budget_policy, _CustomBudgetPolicy)


def test_missing_entry_point_id_fails_at_composition() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_budget_allocation_policy_id = "missing"

    with pytest.raises(ExecutionBudgetAllocationPolicyConfigurationError, match="not found"):
        build_nexus_loop_from_environment(
            AgentRegistry(),
            env=_minimal_env(),
            runtime_config=config,
        )


def test_default_runtime_config_uses_default_shared_pool_policy() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock())

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert isinstance(_child_runner(loop)._budget_policy, DefaultSharedPoolBudgetPolicy)


def test_composition_stores_ledger_factory_not_mutable_ledger() -> None:
    loop_a = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        run_budget=RunBudget(max_tool_calls=10),
    )
    loop_b = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        run_budget=RunBudget(max_tool_calls=10),
    )

    assert loop_a._execution_budget_ledger_factory is not loop_b._execution_budget_ledger_factory
    assert _child_runner(loop_a)._ledger is None
    assert _child_runner(loop_b)._ledger is None


def test_policy_resolved_once_at_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _tracking_load(policy_id: str) -> _CustomBudgetPolicy:
        calls.append(policy_id)
        return _CustomBudgetPolicy()

    monkeypatch.setattr(
        "intergrax.runtime.execution.budget.registry.load_execution_budget_allocation_policy",
        _tracking_load,
    )
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_budget_allocation_policy_id = "custom"

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert calls == ["custom"]
    assert isinstance(_child_runner(loop)._budget_policy, _CustomBudgetPolicy)
