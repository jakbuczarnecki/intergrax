# © Artur Czarnecki. All rights reserved.

"""CAND-006: declarative policy runtime wiring through standard host composition."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from intergrax.applications._shared.policy_wiring import (
    build_runtime_policy_bundle,
    wire_policy_bundle,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import EP_POLICY_RULES, reset_entry_point_spec_cache_for_tests
from intergrax.runtime.policy.policy_bundle import DeclarativePolicyRuntime, RuntimePolicyBundle
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.schema import PolicyRuleAction

pytestmark = pytest.mark.unit

_GROUP = "intergrax.policy_rules"
_INLINE_RULE = {
    "rule_id": "deny_tool",
    "resource_kind": "tool",
    "resource_id": "blocked",
    "action": "deny",
}


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

    def evaluate(
        self,
        rule: object,
        *,
        context: PolicyEvaluationContext,
    ) -> object:
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


def _profile() -> PolicyRulesProfile:
    return PolicyRulesProfile(inline_rules=[_INLINE_RULE])


def test_runtime_policy_bundle_defaults_declarative_runtime_none() -> None:
    bundle = RuntimePolicyBundle()
    assert bundle.declarative_policy_runtime is None


def test_configured_policy_rules_create_declarative_runtime() -> None:
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=False,
    )
    runtime = bundle.declarative_policy_runtime
    assert isinstance(runtime, DeclarativePolicyRuntime)
    assert len(runtime.rules) == 1
    assert runtime.rules[0].rule_id == "deny_tool"


def test_rules_stored_as_immutable_tuple() -> None:
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=False,
    )
    assert bundle.declarative_policy_runtime is not None
    assert isinstance(bundle.declarative_policy_runtime.rules, tuple)


def test_registry_reachable_through_typed_field() -> None:
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=False,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert "deny_tool" in runtime.registry._handlers


def test_no_policy_rules_no_declarative_runtime_even_with_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler")])
    bundle = build_runtime_policy_bundle(discover_entry_points=True)
    assert bundle.declarative_policy_runtime is None


def test_discovery_disabled_no_external_ep_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler")])
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=False,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert "alpha-rule" not in runtime.registry._handlers
    assert runtime.load_report.group == EP_POLICY_RULES
    assert runtime.load_report.registered_count == 0
    assert runtime.load_report.critical_bootstrap_acceptable is True


def test_discovery_enabled_loads_external_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler")])
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=True,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert "alpha-rule" in runtime.registry._handlers
    assert runtime.load_report.registered_count == 1
    assert [item.name for item in runtime.load_report.accepted] == ["alpha"]


def test_broken_ep_isolate_preserves_healthy_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaHandler"),
            _EntryPoint("broken", "not-a-valid-target", _GROUP),
        ],
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=True,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert "alpha-rule" in runtime.registry._handlers
    assert [item.spec.name for item in runtime.load_report.failed] == ["broken"]
    assert runtime.load_report.critical_bootstrap_acceptable is False


def test_invalid_handler_recorded_in_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaHandler"),
            _ep("nope", "_NotAHandler"),
        ],
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=True,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert "nope-rule" not in runtime.registry._handlers
    assert runtime.load_report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    assert runtime.load_report.critical_bootstrap_acceptable is False


def test_budget_reconstruction_preserves_declarative_runtime() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.budget")
    env.policy_rules = _profile()
    bundle = wire_policy_bundle(env)
    assert bundle.budget is not None
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert len(runtime.rules) == 1


def test_legacy_domain_fragments_do_not_store_policy_runtime() -> None:
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=False,
    )
    assert "policy_rules" not in bundle.domain_fragments
    assert "policy_rule_registry" not in bundle.domain_fragments


def test_wire_policy_bundle_standard_host_contract_with_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler")])
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e")
    env.policy_rules = _profile()
    bundle = wire_policy_bundle(env)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert "alpha-rule" in runtime.registry._handlers
    assert runtime.load_report.registered_count == 1


def test_configured_policy_rules_include_mode_and_provenance() -> None:
    bundle = build_runtime_policy_bundle(
        policy_rules=PolicyRulesProfile(
            inline_rules=[_INLINE_RULE],
            policy_enforcement_mode="enforce",
        ),
        discover_entry_points=False,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.enforcement_mode.value == "enforce"
    assert runtime.provenance.rules_digest_sha256


def test_no_evaluate_rule_in_policy_wiring_module() -> None:
    import intergrax.applications._shared.policy_wiring as module

    source = module.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    assert "evaluate_rule" not in text


def test_no_declarative_policy_enforcer_symbol_in_policy_wiring() -> None:
    import intergrax.applications._shared.policy_wiring as module

    source = module.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    assert "DeclarativePolicyEnforcer" not in text
