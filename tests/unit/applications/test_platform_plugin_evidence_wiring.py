# © Artur Czarnecki. All rights reserved.

"""APP-ADOPTION-1: Tier-3 application platform plugin evidence wiring."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.applications._shared import environment_wiring as environment_wiring_module
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
    PolicyRulesProfile,
)
from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_CONTEXT,
    PLATFORM_PLUGIN_DOMAIN_MEMORY,
    PLATFORM_PLUGIN_DOMAIN_POLICY,
)
from intergrax.core.plugins.discovery import EP_CONTEXT, EP_MEMORY_STORES, EP_POLICY_RULES
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.integrations.registry.profile import IntegrationProfile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_INLINE_POLICY_RULE = {
    "rule_id": "evidence.blocked",
    "handler_id": "deny_tool",
    "resource_kind": "tool",
    "resource_id": "blocked",
    "action": "deny",
}
_POLICY_GROUP = "intergrax.policy_rules"


class _PolicyAlphaHandler:
    rule_id = "alpha-rule"

    def evaluate(self, rule: object, *, context: object) -> object:
        from intergrax.runtime.policy.rules.schema import PolicyRuleAction

        return PolicyRuleAction.ALLOW


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


def _dev_legal_settings() -> LegalBackendSettings:
    from intergrax.fastapi_core.config import ApiEnvironment

    return LegalBackendSettings(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        route_prefix="/v1/legal",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=None,
        session_sqlite_path=None,
    )


@pytest.mark.no_ci
def test_wire_application_environment_exposes_memory_and_context_domain_reports() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.memory")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    memory_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_MEMORY)
    assert memory_report is not None
    assert memory_report.group == EP_MEMORY_STORES
    assert memory_report is wiring.platform_plugin_evidence.memory_report()

    context_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_CONTEXT)
    assert context_report is not None
    assert context_report.group == EP_CONTEXT


@pytest.mark.no_ci
def test_wire_application_environment_context_report_matches_same_bootstrap_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []
    original = environment_wiring_module.bootstrap_application_context_catalog

    def _capture(**kwargs: object) -> object:
        result = original(**kwargs)
        captured.append(result.load_report)
        return result

    monkeypatch.setattr(
        environment_wiring_module,
        "bootstrap_application_context_catalog",
        _capture,
    )
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.context-same-pass")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    assert captured
    assert wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_CONTEXT) is captured[0]


@pytest.mark.no_ci
def test_wire_application_environment_memory_report_matches_same_discovery_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []
    original = resolve_memory_platform_wiring

    def _capture(env: ApplicationEnvironmentProfile, **kwargs: object) -> object:
        result = original(env, **kwargs)
        captured.append(result.memory_store_plugin_load_report)
        return result

    monkeypatch.setattr(environment_wiring_module, "resolve_memory_platform_wiring", _capture)
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.same-pass")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    assert captured
    assert wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_MEMORY) is captured[0]


@pytest.mark.no_ci
def test_platform_plugin_evidence_mapping_is_immutable() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.immutable")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    with pytest.raises(TypeError):
        wiring.platform_plugin_evidence.domain_reports["memory"] = wiring.platform_plugin_evidence.memory_report()


@pytest.mark.no_ci
def test_baseline_environment_has_deterministic_empty_memory_and_context_evidence() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.baseline")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    memory_report = wiring.platform_plugin_evidence.memory_report()
    assert memory_report.accepted == ()
    assert memory_report.rejected == ()
    assert memory_report.failed == ()
    assert memory_report.registered_count == 0

    context_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_CONTEXT)
    assert context_report is not None
    assert context_report.group == EP_CONTEXT
    assert context_report.accepted == ()
    assert context_report.rejected == ()
    assert context_report.failed == ()
    assert context_report.registered_count == 0


@pytest.mark.no_ci
def test_policy_domain_report_propagates_without_second_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "alpha",
                f"{__name__}:_PolicyAlphaHandler",
                _POLICY_GROUP,
            )
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.policy")
    env = env.model_copy(
        update={
            "policy_rules": PolicyRulesProfile(inline_rules=[_INLINE_POLICY_RULE]),
        },
    )
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        conformance_check=False,
    )

    policy_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_POLICY)
    assert policy_report is not None
    runtime = wiring.policy_bundle.declarative_policy_runtime
    assert runtime is not None
    assert policy_report is runtime.load_report
    assert policy_report.group == EP_POLICY_RULES


@pytest.mark.no_ci
def test_policy_domain_absent_when_policy_rules_not_configured() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.no-policy")
    env = env.model_copy(update={"policy_rules": None})
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    assert wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_POLICY) is None


@pytest.mark.no_ci
def test_failed_non_selected_memory_ep_in_evidence_without_failing_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_profile_ep = (
        "tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin:"
        "ExternalInMemoryUserProfileStorePlugin"
    )
    entries = _EntryPoints(
        [
            _EntryPoint("external_user_profile", user_profile_ep, EP_MEMORY_STORES),
            _EntryPoint("broken_sibling", "not-a-valid-target", EP_MEMORY_STORES),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ppe.failed-sibling")
    env = env.model_copy(
        update={
            "integration_profile": IntegrationProfile(),
            "memory_profile": MemoryProfile(
                user_profile_store_plugin_id="external.in_memory_user_profile",
            ),
        },
    )
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        conformance_check=False,
    )

    memory_report = wiring.platform_plugin_evidence.memory_report()
    assert len(memory_report.failed) == 1
    assert wiring.tool_wiring is not None


@pytest.mark.no_ci
def test_legal_host_wires_with_platform_plugin_evidence() -> None:
    settings = _dev_legal_settings()
    manifest = build_legal_manifest(settings)
    env = build_legal_environment_profile(settings)
    wiring = wire_application_environment(manifest, env, settings=settings, conformance_check=False)

    assert wiring.platform_plugin_evidence.memory_report().group == EP_MEMORY_STORES
    assert wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_CONTEXT) is not None
