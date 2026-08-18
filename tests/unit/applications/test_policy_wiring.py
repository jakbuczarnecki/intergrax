# © Artur Czarnecki. All rights reserved.

"""CAND-006: declarative policy runtime wiring through standard host composition."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.applications._shared.policy_wiring import (
    build_runtime_policy_bundle,
    wire_policy_bundle,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_POLICY_DEFINITIONS,
    EP_POLICY_RULES,
    EntryPointSpec,
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
from intergrax.core.distribution import (
    DistributionPackageIdentity,
    PlatformCompatibility,
    check_platform_compatibility,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from intergrax.contracts.tool_invocation_control_policy import (
    TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
)
from intergrax.runtime.policy.builtin_catalog import (
    TOOL_INVOCATION_CONTROL_POLICY_ID,
    TOOL_INVOCATION_CONTROL_VERSION,
)
from intergrax.runtime.policy.catalog import (
    PolicyDefinitionConflictError,
    UnknownPolicyDefinitionError,
)
from intergrax.runtime.policy.configuration_contract import ConfigurationContractBinding
from intergrax.runtime.policy.contribution import GovernancePolicyContribution
from intergrax.runtime.policy.policy_bundle import DeclarativePolicyRuntime, RuntimePolicyBundle
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.runtime.policy.rules.schema import PolicyRuleAction

pytestmark = pytest.mark.unit

_GROUP = "intergrax.policy_rules"
_DEFINITIONS_GROUP = EP_POLICY_DEFINITIONS
_PACKAGE_NAME = "alpha-policy-plugin"
_PACKAGE_VERSION = "1.0.0"
_OTHER_PACKAGE_NAME = "beta-policy-plugin"
_HANDLER_ID = "alpha-rule"
_POLICY_ID = "data_export_control"
_POLICY_VERSION = "2"
_CONFIG_CONTRACT_ID = "acme.data_export_control.v1"
_PLATFORM_VERSION = "0.1.0"


class _PluginTestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True


_PLUGIN_BINDING = ConfigurationContractBinding.from_pydantic_model(
    _CONFIG_CONTRACT_ID,
    _PluginTestConfig,
)
_INLINE_RULE = {
    "rule_id": "wiring.blocked",
    "handler_id": "deny_tool",
    "resource_kind": "tool",
    "resource_id": "blocked",
    "action": "deny",
}


class _Dist:
    def __init__(self, name: str) -> None:
        self.name = name


class _EntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        group: str,
        *,
        distribution: str | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.group = group
        self.dist = _Dist(distribution) if distribution is not None else None


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


def _definition_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        _DEFINITIONS_GROUP,
        distribution=distribution,
    )


def _rule_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        _GROUP,
        distribution=distribution,
    )


def _plugin_policy_definition(**overrides: object) -> PolicyDefinition:
    payload: dict[str, object] = {
        "policy_id": _POLICY_ID,
        "version": _POLICY_VERSION,
        "display_name": "Data export control",
        "handler_id": _HANDLER_ID,
        "configuration_contract_id": _CONFIG_CONTRACT_ID,
        "source": PolicyDefinitionSource.PLUGIN,
    }
    payload.update(overrides)
    return PolicyDefinition.model_validate(payload)


_CONTRIBUTION = _plugin_policy_definition()


def _governance_contribution(
    definition: PolicyDefinition | None = None,
    *,
    configuration_contract_binding: ConfigurationContractBinding | None = _PLUGIN_BINDING,
) -> GovernancePolicyContribution:
    return GovernancePolicyContribution(
        definition=definition or _CONTRIBUTION,
        package_identity=DistributionPackageIdentity(
            name=_PACKAGE_NAME,
            version=_PACKAGE_VERSION,
        ),
        configuration_contract_binding=configuration_contract_binding,
    )


_CONTRIBUTION_EP = _governance_contribution()


def _conflicting_contribution() -> GovernancePolicyContribution:
    return _governance_contribution(
        _plugin_policy_definition(
            policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
            version=TOOL_INVOCATION_CONTROL_VERSION,
            handler_id=_HANDLER_ID,
            configuration_contract_id=_CONFIG_CONTRACT_ID,
        ),
        configuration_contract_binding=_PLUGIN_BINDING,
    )


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
                ref="tests/unit/applications/test_policy_wiring.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )


def _lookup_for(qualification: object | None) -> object:
    return lambda spec: qualification


def _lookup_for_packages(qualifications: dict[str, object | None]) -> object:
    def lookup(spec: EntryPointSpec) -> object | None:
        if spec.distribution is None:
            return None
        return qualifications.get(spec.distribution)

    return lookup


def _qualification_bundle(qualification: object) -> PlatformPluginPackageQualificationBundle:
    identity = DistributionPackageIdentity(name=_PACKAGE_NAME, version=_PACKAGE_VERSION)
    return PlatformPluginPackageQualificationBundle([(identity, qualification)])


def _mock_installed_distribution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    package_name: str = _PACKAGE_NAME,
    package_version: str = _PACKAGE_VERSION,
    extra_packages: dict[str, str] | None = None,
) -> None:
    versions = {package_name: package_version}
    if extra_packages:
        versions.update(extra_packages)

    def _make_dist(version: str) -> MagicMock:
        dist = MagicMock()
        dist.version = version
        dist.files = None
        return dist

    intergrax_dist = MagicMock()
    intergrax_dist.version = _PLATFORM_VERSION

    def _distribution(name: str) -> MagicMock:
        if name in versions:
            return _make_dist(versions[name])
        if name in ("intergrax", "Intergrax-ai"):
            return intergrax_dist
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", _distribution)


def _strict_profile() -> PolicyRulesProfile:
    return PolicyRulesProfile(inline_rules=[_INLINE_RULE])


def _profile() -> PolicyRulesProfile:
    return PolicyRulesProfile(inline_rules=[_INLINE_RULE])


def test_runtime_policy_bundle_defaults_declarative_runtime_none() -> None:
    bundle = RuntimePolicyBundle()
    assert bundle.declarative_policy_runtime is None
    resolved = bundle.policy_catalog.resolve(
        policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
        version=TOOL_INVOCATION_CONTROL_VERSION,
    )
    assert resolved.source is PolicyDefinitionSource.BUILT_IN


def test_configured_policy_rules_create_declarative_runtime() -> None:
    bundle = build_runtime_policy_bundle(
        policy_rules=_profile(),
        discover_entry_points=False,
    )
    runtime = bundle.declarative_policy_runtime
    assert isinstance(runtime, DeclarativePolicyRuntime)
    assert len(runtime.rules) == 1
    assert runtime.rules[0].rule_id == "wiring.blocked"
    assert runtime.rules[0].handler_id == "deny_tool"


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


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        (None, PolicyEnforcementMode.AUDIT_ONLY),
        ("audit_only", PolicyEnforcementMode.AUDIT_ONLY),
        ("enforce", PolicyEnforcementMode.ENFORCE),
    ],
)
def test_policy_enforcement_mode_valid_values(
    raw_mode: str | None,
    expected: PolicyEnforcementMode,
) -> None:
    if raw_mode is None:
        profile = PolicyRulesProfile()
    else:
        profile = PolicyRulesProfile(policy_enforcement_mode=raw_mode)
    assert profile.policy_enforcement_mode is expected
    bundle = build_runtime_policy_bundle(policy_rules=profile, discover_entry_points=False)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.enforcement_mode is expected


def test_policy_enforcement_mode_typo_fails_validation() -> None:
    with pytest.raises(ValidationError):
        PolicyRulesProfile(policy_enforcement_mode="enfroce")


def test_policy_rules_none_has_builtin_catalog_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    bundle = build_runtime_policy_bundle(discover_entry_points=True)
    assert bundle.declarative_policy_runtime is None
    resolved = bundle.policy_catalog.resolve(
        policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
        version=TOOL_INVOCATION_CONTROL_VERSION,
    )
    assert resolved.source is PolicyDefinitionSource.BUILT_IN
    with pytest.raises(UnknownPolicyDefinitionError):
        bundle.policy_catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)


def test_canonical_bundle_resolves_qualified_plugin_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_strict_profile(),
        execution_mode=ExecutionMode.STRICT,
        discover_entry_points=True,
        package_qualification_lookup=_lookup_for(qualification),
    )
    resolved = bundle.policy_catalog.resolve(
        policy_id=_POLICY_ID,
        version=_POLICY_VERSION,
    )
    assert resolved.source is PolicyDefinitionSource.PLUGIN
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is not None


def test_canonical_bundle_strict_unqualified_plugin_not_in_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_strict_profile(),
        execution_mode=ExecutionMode.STRICT,
        discover_entry_points=True,
        package_qualification_lookup=_lookup_for(None),
    )
    with pytest.raises(UnknownPolicyDefinitionError):
        bundle.policy_catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)


def test_canonical_bundle_handler_provenance_mismatch_excludes_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    alpha_qualification = _production_package_qualification(compatibility=compatibility)
    beta_qualification = build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=_OTHER_PACKAGE_NAME,
            package_version=_PACKAGE_VERSION,
            domain="policy",
            entry_point_group=EP_POLICY_RULES,
            entry_point_name="beta",
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            compatibility_evidence(compatibility),
            QualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="policy.tests.passed",
                ref="tests/unit/applications/test_policy_wiring.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep(
                "beta-policy",
                "_CONTRIBUTION",
                distribution=_OTHER_PACKAGE_NAME,
            ),
        ],
    )
    _mock_installed_distribution(
        monkeypatch,
        extra_packages={_OTHER_PACKAGE_NAME: _PACKAGE_VERSION},
    )
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_strict_profile(),
        execution_mode=ExecutionMode.STRICT,
        discover_entry_points=True,
        package_qualification_lookup=_lookup_for_packages(
            {
                _PACKAGE_NAME: alpha_qualification,
                _OTHER_PACKAGE_NAME: beta_qualification,
            }
        ),
    )
    with pytest.raises(UnknownPolicyDefinitionError):
        bundle.policy_catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)


def test_canonical_bundle_plugin_builtin_conflict_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_conflicting_contribution", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    with pytest.raises(PolicyDefinitionConflictError):
        build_runtime_policy_bundle(
            policy_rules=_strict_profile(),
            execution_mode=ExecutionMode.STRICT,
            discover_entry_points=True,
            package_qualification_lookup=_lookup_for(qualification),
        )


def test_discover_plugins_false_builtin_catalog_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_strict_profile(),
        discover_entry_points=False,
    )
    with pytest.raises(UnknownPolicyDefinitionError):
        bundle.policy_catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)
    resolved = bundle.policy_catalog.resolve(
        policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
        version=TOOL_INVOCATION_CONTROL_VERSION,
    )
    assert resolved.source is PolicyDefinitionSource.BUILT_IN


def test_budget_reconstruction_preserves_policy_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.catalog.budget")
    env.policy_rules = _strict_profile()
    env = env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
        },
    )
    qualifications = _qualification_bundle(qualification)
    base = build_runtime_policy_bundle(
        policy_rules=env.policy_rules,
        execution_mode=env.execution_mode,
        discover_entry_points=True,
        package_qualification_lookup=qualifications.lookup_for_entry_point,
    )
    expected = base.policy_catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    bundle = wire_policy_bundle(env, package_qualifications=qualifications)
    assert bundle.budget is not None
    assert bundle.policy_catalog.resolve(
        policy_id=_POLICY_ID,
        version=_POLICY_VERSION,
    ) == expected


def test_handler_discovery_occurs_once_per_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    calls = {"count": 0}
    original = __import__(
        "intergrax.runtime.policy.rules.plugin_loader",
        fromlist=["load_policy_rule_plugin_report"],
    ).load_policy_rule_plugin_report

    def _counting_loader(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "intergrax.applications._shared.policy_wiring.load_policy_rule_plugin_report",
        _counting_loader,
    )
    build_runtime_policy_bundle(
        policy_rules=_strict_profile(),
        execution_mode=ExecutionMode.STRICT,
        discover_entry_points=True,
        package_qualification_lookup=_lookup_for(qualification),
    )
    assert calls["count"] == 1
