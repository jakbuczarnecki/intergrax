# © Artur Czarnecki. All rights reserved.

"""UE-8P2 — execution authority policy resolver and entry-point registry tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    ChildAuthorityResolution,
    DefaultStrictAuthorityPolicy,
    ExecutionAuthorityPolicy,
)
from intergrax.runtime.execution.authority.registry import (
    ExecutionAuthorityPolicyConfigurationError,
    load_execution_authority_policy,
    resolve_execution_authority_policy,
    resolve_execution_authority_policy_from_runtime_config,
)
from intergrax.runtime.nexus.config import RuntimeConfig

pytestmark = pytest.mark.unit


class _EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        return self._value


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


class _InvalidAuthorityPolicy:
    pass


def test_resolve_prefers_explicit_instance() -> None:
    custom = _CustomAuthorityPolicy()
    resolved = resolve_execution_authority_policy(
        policy_override=custom,
        entry_point_policy_id="would-fail-if-used",
    )
    assert resolved is custom


def test_resolve_without_configuration_returns_default() -> None:
    resolved = resolve_execution_authority_policy()
    assert isinstance(resolved, DefaultStrictAuthorityPolicy)


def test_resolve_loads_valid_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.runtime.execution.authority.registry.entry_points",
        lambda group=None: [_EntryPoint("custom", _CustomAuthorityPolicy)],
    )
    resolved = resolve_execution_authority_policy(entry_point_policy_id="custom")
    assert isinstance(resolved, _CustomAuthorityPolicy)


def test_load_execution_authority_policy_instantiates_class_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "intergrax.runtime.execution.authority.registry.entry_points",
        lambda group=None: [_EntryPoint("custom", _CustomAuthorityPolicy)],
    )
    loaded = load_execution_authority_policy("custom")
    assert isinstance(loaded, _CustomAuthorityPolicy)


def test_resolve_missing_entry_point_fails_closed() -> None:
    with pytest.raises(ExecutionAuthorityPolicyConfigurationError, match="not found"):
        resolve_execution_authority_policy(entry_point_policy_id="missing-policy")


def test_load_invalid_entry_point_object_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "intergrax.runtime.execution.authority.registry.entry_points",
        lambda group=None: [_EntryPoint("bad", _InvalidAuthorityPolicy)],
    )
    with pytest.raises(TypeError, match="ExecutionAuthorityPolicy"):
        load_execution_authority_policy("bad")


def test_resolve_from_runtime_config_prefers_instance_override() -> None:
    custom = _CustomAuthorityPolicy()
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_authority_policy = custom
    config.execution_authority_policy_id = "ignored"
    resolved = resolve_execution_authority_policy_from_runtime_config(config)
    assert resolved is custom


def test_entry_point_group_name() -> None:
    from intergrax.runtime.execution.authority import registry

    assert registry._ENTRY_POINT_GROUP == "intergrax.execution_authority_policies"
