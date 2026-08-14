# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.core.plugins.errors import PluginLoadError
from intergrax.runtime.policy.rules.plugin_loader import (
    PolicyRuleLoadPolicy,
    load_policy_rule_plugin_report,
    load_policy_rule_plugins,
)
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import PolicyRuleAction

pytestmark = pytest.mark.unit

_GROUP = "intergrax.policy_rules"


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


class _AlphaHandler:
    rule_id = "alpha-rule"

    def evaluate(self, rule: object, *, context: dict[str, str]) -> object:
        return PolicyRuleAction.ALLOW


class _BetaHandler:
    rule_id = "beta-rule"

    def evaluate(self, rule: object, *, context: dict[str, str]) -> object:
        return PolicyRuleAction.ALLOW


class _NotAHandler:
    pass


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", _GROUP)


def test_valid_handler_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler")])
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(registry)
    assert report.registered_count == 1
    assert [item.name for item in report.accepted] == ["alpha"]
    assert "alpha-rule" in registry._handlers
    assert "deny_tool" in registry._handlers


def test_broken_and_valid_handler_isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaHandler"),
            _EntryPoint("broken", "not-a-valid-target", _GROUP),
            _ep("beta", "_BetaHandler"),
        ],
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(on_load_failure="isolate"),
    )
    assert [item.name for item in report.accepted] == ["alpha", "beta"]
    assert [item.spec.name for item in report.failed] == ["broken"]
    assert "alpha-rule" in registry._handlers
    assert "beta-rule" in registry._handlers
    assert "deny_tool" in registry._handlers


def test_fail_fast_preserves_legacy_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaHandler"),
            _EntryPoint("broken", "not-a-valid-target", _GROUP),
        ],
    )
    registry = PolicyRuleRegistry()
    with pytest.raises(PluginLoadError):
        load_policy_rule_plugins(registry)


def test_invalid_handler_target_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaHandler"),
            _ep("nope", "_NotAHandler"),
        ],
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(registry)
    assert report.registered_count == 1
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    assert report.rejected[0].spec.name == "nope"
    assert "alpha-rule" in registry._handlers


def test_int_compatibility_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler")])
    registry = PolicyRuleRegistry()
    count = load_policy_rule_plugins(registry)
    assert isinstance(count, int)
    assert count == 1
