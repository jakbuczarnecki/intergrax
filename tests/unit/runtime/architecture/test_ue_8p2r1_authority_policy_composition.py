# © Artur Czarnecki. All rights reserved.

"""UE-8P2R1 — RuntimeConfig authority policy reaches GraphExecutor child runner."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.core.plugins.discovery import (
    EP_EXECUTION_AUTHORITY_POLICIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    ChildAuthorityResolution,
    DefaultStrictAuthorityPolicy,
)
from intergrax.runtime.execution.authority.registry import (
    ExecutionAuthorityPolicyConfigurationError,
)
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
    return _EntryPoint(name, f"{__name__}:{attr}", EP_EXECUTION_AUTHORITY_POLICIES)


class _CustomAuthorityPolicy:
    def resolve_child_authority(
        self,
        context: ChildAuthorityContext,
    ) -> ChildAuthorityResolution:
        _ = context
        return ChildAuthorityResolution(
            authority=ParentExecutionAuthority.scoped(("custom",)),
            effective_delegation=None,
        )


def _minimal_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults()


def _child_runner_policy(loop: object) -> object:
    return loop._graph_executor._child_runner._authority_policy  # noqa: SLF001


def test_runtime_config_direct_instance_reaches_graph_executor_child_runner() -> None:
    custom = _CustomAuthorityPolicy()
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_authority_policy = custom

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert _child_runner_policy(loop) is custom


def test_runtime_config_entry_point_id_reaches_graph_executor_child_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_policy_ep("custom", "_CustomAuthorityPolicy")])
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_authority_policy_id = "custom"

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert isinstance(_child_runner_policy(loop), _CustomAuthorityPolicy)


def test_missing_entry_point_id_fails_at_composition() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_authority_policy_id = "missing"

    with pytest.raises(ExecutionAuthorityPolicyConfigurationError, match="not found"):
        build_nexus_loop_from_environment(
            AgentRegistry(),
            env=_minimal_env(),
            runtime_config=config,
        )


def test_default_runtime_config_uses_default_strict_authority_policy() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock())

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert isinstance(_child_runner_policy(loop), DefaultStrictAuthorityPolicy)


def test_no_runtime_config_uses_default_strict_authority_policy() -> None:
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
    )

    assert isinstance(_child_runner_policy(loop), DefaultStrictAuthorityPolicy)


def test_policy_resolved_once_at_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _tracking_load(policy_id: str) -> _CustomAuthorityPolicy:
        calls.append(policy_id)
        return _CustomAuthorityPolicy()

    monkeypatch.setattr(
        "intergrax.runtime.execution.authority.registry.load_execution_authority_policy",
        _tracking_load,
    )
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_authority_policy_id = "custom"

    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_minimal_env(),
        runtime_config=config,
    )

    assert calls == ["custom"]
    assert isinstance(_child_runner_policy(loop), _CustomAuthorityPolicy)
