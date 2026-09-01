# © Artur Czarnecki. All rights reserved.

"""PLUGIN-ENGINE-CROSS-FLOW-1: end-to-end application plugin host wiring proof."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.applications._shared import environment_wiring as environment_wiring_module
from intergrax.applications._shared.context_wiring import ContextAssemblyError
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.context_wiring import resolve_context_plugin_registry_from_environment
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
    MemoryProfile,
    PolicyRulesProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_CONTEXT,
    PLATFORM_PLUGIN_DOMAIN_MEMORY,
    PLATFORM_PLUGIN_DOMAIN_POLICY,
    PLATFORM_PLUGIN_DOMAIN_SECURITY,
)
from intergrax.core.catalog_bootstrap import reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.distribution import DistributionPackageIdentity, PlatformCompatibility, check_platform_compatibility
from intergrax.core.plugin_env import INTERGRAX_DISCOVER_PLUGINS_ENV
from intergrax.core.plugins.discovery import (
    EP_CONTEXT,
    EP_MEMORY_STORES,
    EP_POLICY_DEFINITIONS,
    EP_POLICY_RULES,
    EP_SECURITY_DEFENSES,
    EP_TOOLS,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.core.plugins.platform_qualification import (
    PlatformPluginPackageQualificationBundle,
    PluginQualificationEvidenceKind,
    PluginQualificationLevel,
    build_external_package_subject,
    build_qualification_result,
    compatibility_evidence,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.context.bootstrap import reset_context_catalog_bootstrap_for_tests
from intergrax.context.registry import clear_context_plugin_catalog
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.policy.contribution import GovernancePolicyContribution
from intergrax.runtime.policy.configuration_contract import ConfigurationContractBinding
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.schema import PolicyRuleAction
from intergrax.runtime.security.defense_registry import get_security_defense_plugin, reset_security_defense_registry_for_tests
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.profile import ToolProfile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter
from tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin import (
    FixtureExternalUserProfileStore,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gate,
    pytest.mark.usefixtures(
        "catalog_fixture_installed",
        "security_defense_fixture_installed",
        "reference_enterprise_plugin_installed",
    ),
]

_POLICY_GROUP = EP_POLICY_RULES
_DEFINITIONS_GROUP = EP_POLICY_DEFINITIONS
_PACKAGE_NAME = "alpha-policy-plugin"
_PACKAGE_VERSION = "1.0.0"
_HANDLER_ID = "alpha-rule"
_POLICY_ID = "data_export_control"
_POLICY_VERSION = "2"
_CONFIG_CONTRACT_ID = "acme.data_export_control.v1"
_PLATFORM_VERSION = "0.1.0"
_MEMORY_USER_PROFILE_EP = (
    "tests.fixtures.plugin_packages.memory_store_plugin.memory_store_plugin.plugin:"
    "ExternalInMemoryUserProfileStorePlugin"
)
_FIXTURE_ECHO_TOOL_ID = "fixture_ep.echo"
_INLINE_POLICY_RULE = {
    "rule_id": "crossflow.blocked",
    "handler_id": "deny_tool",
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


class _PluginTestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True


_PLUGIN_BINDING = ConfigurationContractBinding.from_pydantic_model(
    _CONFIG_CONTRACT_ID,
    _PluginTestConfig,
)


class _AlphaHandler:
    rule_id = "alpha-rule"

    def evaluate(
        self,
        rule: object,
        *,
        context: PolicyEvaluationContext,
    ) -> object:
        _ = rule, context
        return PolicyRuleAction.ALLOW


class _UnsupportedContextTarget:
    value = "not-a-plugin"


def _plugin_policy_definition() -> PolicyDefinition:
    return PolicyDefinition.model_validate(
        {
            "policy_id": _POLICY_ID,
            "version": _POLICY_VERSION,
            "display_name": "Data export control",
            "handler_id": _HANDLER_ID,
            "configuration_contract_id": _CONFIG_CONTRACT_ID,
            "source": PolicyDefinitionSource.PLUGIN,
        },
    )


_CONTRIBUTION_EP = GovernancePolicyContribution(
    definition=_plugin_policy_definition(),
    package_identity=DistributionPackageIdentity(
        name=_PACKAGE_NAME,
        version=_PACKAGE_VERSION,
    ),
    configuration_contract_binding=_PLUGIN_BINDING,
)


@pytest.fixture(autouse=True)
def _reset_plugin_state() -> None:
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    reset_security_defense_registry_for_tests()
    reset_context_catalog_bootstrap_for_tests()
    clear_context_plugin_catalog()
    yield
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    reset_security_defense_registry_for_tests()
    reset_context_catalog_bootstrap_for_tests()
    clear_context_plugin_catalog()


@pytest.fixture(autouse=True)
def _stub_environment_llm_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        environment_wiring_module,
        "resolve_environment_llm_adapter",
        lambda _env: FakeLLMAdapter(),
    )


def _install_cross_flow_eps(
    monkeypatch: pytest.MonkeyPatch,
    extra: list[_EntryPoint] | None = None,
) -> None:
    module = __name__
    entries = [
        _EntryPoint(
            "fixture_echo",
            "intergrax_catalog_fixture.tool:FixtureEchoToolPlugin",
            EP_TOOLS,
        ),
        _EntryPoint(
            "fixture_defense",
            "intergrax_security_defense_fixture.plugin:FixtureDefensePlugin",
            EP_SECURITY_DEFENSES,
        ),
        _EntryPoint(
            "reference_enterprise",
            "intergrax_reference_enterprise_plugin.context:ReferenceEnterpriseContextPlugin",
            EP_CONTEXT,
        ),
        _EntryPoint("alpha", f"{module}:_AlphaHandler", _POLICY_GROUP),
        _EntryPoint("alpha-policy", f"{module}:_CONTRIBUTION_EP", _DEFINITIONS_GROUP),
        _EntryPoint("external_user_profile", _MEMORY_USER_PROFILE_EP, EP_MEMORY_STORES),
    ]
    if extra:
        entries.extend(extra)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _compatible_platform() -> object:
    return check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=0.1,<2"),
        _PLATFORM_VERSION,
    )


def _production_package_qualification(*, compatibility: object) -> object:
    return build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=_PACKAGE_NAME,
            package_version=_PACKAGE_VERSION,
            domain="policy",
            entry_point_group=EP_POLICY_RULES,
            entry_point_name="alpha",
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            compatibility_evidence(compatibility),
            QualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="policy.tests.passed",
                ref="tests/integration/platform_plugins/test_plugin_engine_cross_flow.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )


def _qualification_bundle(qualification: object) -> PlatformPluginPackageQualificationBundle:
    identity = DistributionPackageIdentity(name=_PACKAGE_NAME, version=_PACKAGE_VERSION)
    return PlatformPluginPackageQualificationBundle([(identity, qualification)])


def _mock_installed_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _make_dist(version: str) -> MagicMock:
        dist = MagicMock()
        dist.version = version
        dist.files = None
        return dist

    intergrax_dist = MagicMock()
    intergrax_dist.version = _PLATFORM_VERSION

    def _distribution(name: str) -> MagicMock:
        if name == _PACKAGE_NAME:
            return _make_dist(_PACKAGE_VERSION)
        if name in ("intergrax", "Intergrax-ai"):
            return intergrax_dist
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", _distribution)


def _strict_cross_flow_env(profile_id: str) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    return env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
            "policy_rules": PolicyRulesProfile(inline_rules=[_INLINE_POLICY_RULE]),
            "integration_profile": IntegrationProfile(),
            "memory_profile": MemoryProfile(
                user_profile_store_plugin_id="external.in_memory_user_profile",
            ),
            "context_profile": ContextProfile(
                context_plugin_ids=["reference_enterprise.context"],
                enable_rag=False,
                enable_websearch=False,
            ),
            "capabilities": env.capabilities.model_copy(
                update={
                    "tools": ToolProfile(enabled_bundles=["fixture_ep"]),
                },
            ),
        },
    )


def _configure_policy_discovery(monkeypatch: pytest.MonkeyPatch) -> PlatformPluginPackageQualificationBundle:
    compatibility = _compatible_platform()
    qualification = _production_package_qualification(compatibility=compatibility)
    _install_cross_flow_eps(monkeypatch)
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    return _qualification_bundle(qualification)


def test_strict_application_plugin_cross_flow_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    qualifications = _configure_policy_discovery(monkeypatch)
    settings = LabApplicationSettings.from_env()
    env = _strict_cross_flow_env("plugin.crossflow.happy")

    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        conformance_check=False,
        platform_plugin_package_qualifications=qualifications,
    )

    # Tool: canonical external EP path (see test_plugin8_dual_mode_tool_e2e for dual-mode proof).
    assert wiring.tool_wiring.registry.has(_FIXTURE_ECHO_TOOL_ID)

    policy_runtime = wiring.policy_bundle.declarative_policy_runtime
    assert policy_runtime is not None
    assert "alpha-rule" in policy_runtime.registry._handlers

    assert get_security_defense_plugin("fixture_ep.defense") is not None

    context_registry = resolve_context_plugin_registry_from_environment(wiring.profile)
    provider_ids = {provider.provider_id for provider in context_registry.list_providers()}
    assert "reference_enterprise.stub" in provider_ids

    memory_wiring = resolve_memory_platform_wiring(wiring.profile, discover_entry_points=True)
    assert isinstance(memory_wiring.user_profile_store, FixtureExternalUserProfileStore)

    evidence = wiring.platform_plugin_evidence
    security_report = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SECURITY)
    assert security_report is not None
    assert [item.name for item in security_report.accepted] == ["fixture_defense"]
    assert security_report.failed == ()
    assert security_report.rejected == ()
    assert security_report.critical_bootstrap_acceptable is True

    policy_report = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_POLICY)
    assert policy_report is policy_runtime.load_report
    assert [item.name for item in policy_report.accepted] == ["alpha"]
    assert policy_report.failed == ()
    assert policy_report.rejected == ()
    assert policy_report.critical_bootstrap_acceptable is True

    context_report = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_CONTEXT)
    assert context_report is not None
    assert "reference_enterprise" in [item.name for item in context_report.accepted]
    assert context_report.failed == ()
    assert context_report.rejected == ()
    assert context_report.critical_bootstrap_acceptable is True

    memory_report = evidence.report_for(PLATFORM_PLUGIN_DOMAIN_MEMORY)
    assert memory_report is not None
    assert [item.name for item in memory_report.accepted] == ["external_user_profile"]
    assert memory_report.failed == ()
    assert memory_report.rejected == ()
    assert memory_report.critical_bootstrap_acceptable is True


def test_strict_application_plugin_cross_flow_rejects_invalid_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualifications = _configure_policy_discovery(monkeypatch)
    _install_cross_flow_eps(
        monkeypatch,
        extra=[
            _EntryPoint(
                "unsupported_ep",
                f"{__name__}:_UnsupportedContextTarget",
                EP_CONTEXT,
            ),
        ],
    )
    settings = LabApplicationSettings.from_env()
    env = _strict_cross_flow_env("plugin.crossflow.strict-invalid")

    with pytest.raises(ContextAssemblyError, match="invalid_target_type"):
        wire_application_environment(
            build_lab_manifest(settings),
            env,
            conformance_check=False,
            platform_plugin_package_qualifications=qualifications,
        )
