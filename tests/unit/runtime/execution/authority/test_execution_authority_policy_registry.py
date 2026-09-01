# © Artur Czarnecki. All rights reserved.

"""UE-8P2 — execution authority policy resolver and entry-point registry tests."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.core.plugins.discovery import (
    EP_EXECUTION_AUTHORITY_POLICIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    ChildAuthorityResolution,
    DefaultStrictAuthorityPolicy,
    ExecutionAuthorityPolicy,
)
from intergrax.runtime.execution.authority.registry import (
    ExecutionAuthorityPolicyConfigurationError,
    list_execution_authority_policy_ids,
    load_execution_authority_policy,
    resolve_execution_authority_policy,
    resolve_execution_authority_policy_from_runtime_config,
)
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


_CUSTOM_AUTHORITY_POLICY_INSTANCE = _CustomAuthorityPolicy()


class _InvalidAuthorityPolicy:
    pass


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _policy_ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", EP_EXECUTION_AUTHORITY_POLICIES)


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
    _install_eps(monkeypatch, [_policy_ep("custom", "_CustomAuthorityPolicy")])
    resolved = resolve_execution_authority_policy(entry_point_policy_id="custom")
    assert isinstance(resolved, _CustomAuthorityPolicy)


def test_load_execution_authority_policy_instantiates_class_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_policy_ep("custom", "_CustomAuthorityPolicy")])
    loaded = load_execution_authority_policy("custom")
    assert isinstance(loaded, _CustomAuthorityPolicy)


def test_load_instance_entry_point_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_policy_ep("custom", "_CUSTOM_AUTHORITY_POLICY_INSTANCE")])
    loaded = load_execution_authority_policy("custom")
    assert loaded is _CUSTOM_AUTHORITY_POLICY_INSTANCE


def test_load_missing_entry_point_returns_none() -> None:
    assert load_execution_authority_policy("missing-policy") is None


def test_resolve_missing_entry_point_fails_closed() -> None:
    with pytest.raises(ExecutionAuthorityPolicyConfigurationError, match="not found"):
        resolve_execution_authority_policy(entry_point_policy_id="missing-policy")


def test_load_invalid_entry_point_object_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_policy_ep("bad", "_InvalidAuthorityPolicy")])
    with pytest.raises(TypeError, match="ExecutionAuthorityPolicy"):
        load_execution_authority_policy("bad")


def test_list_execution_authority_policy_ids_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _policy_ep("beta", "_CustomAuthorityPolicy"),
            _policy_ep("alpha", "_CustomAuthorityPolicy"),
        ],
    )
    assert list_execution_authority_policy_ids() == ("alpha", "beta")


def test_resolve_from_runtime_config_prefers_instance_override() -> None:
    custom = _CustomAuthorityPolicy()
    config = RuntimeConfig(llm_adapter=MagicMock())
    config.execution_authority_policy = custom
    config.execution_authority_policy_id = "ignored"
    resolved = resolve_execution_authority_policy_from_runtime_config(config)
    assert resolved is custom


def test_entry_point_group_name() -> None:
    from intergrax.core.plugins.discovery import EP_EXECUTION_AUTHORITY_POLICIES
    from intergrax.runtime.execution.authority import registry

    assert registry._ENTRY_POINT_GROUP == EP_EXECUTION_AUTHORITY_POLICIES
