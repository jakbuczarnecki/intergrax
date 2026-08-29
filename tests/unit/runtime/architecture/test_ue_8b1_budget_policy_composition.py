# © Artur Czarnecki. All rights reserved.

"""UE-8B1 — RuntimeConfig budget allocation policy reaches GraphExecutor child runner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
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
    monkeypatch.setattr(
        "intergrax.runtime.execution.budget.registry.entry_points",
        lambda group=None: [_EntryPoint("custom", _CustomBudgetPolicy)],
    )
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


def test_composition_creates_one_canonical_ledger_per_run() -> None:
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

    ledger_a = _child_runner(loop_a)._ledger
    ledger_b = _child_runner(loop_b)._ledger
    assert ledger_a is not ledger_b


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
