# © Artur Czarnecki. All rights reserved.

"""Configuration contract binding and registry tests (G4B-3)."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications.contracts.environment_profile import PolicyRulesProfile
from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.contracts.tool_invocation_control_policy import (
    TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
    ToolInvocationControlConfig,
)
from intergrax.core.distribution import DistributionPackageIdentity, PlatformCompatibility, check_platform_compatibility
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_POLICY_DEFINITIONS,
    EP_POLICY_RULES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.core.plugins.platform_qualification import (
    PluginQualificationEvidenceKind,
    PluginQualificationLevel,
    build_external_package_subject,
    build_qualification_result,
    compatibility_evidence,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from intergrax.runtime.policy.configuration_contract import (
    ConfigurationContractBinding,
    ConfigurationContractConflictError,
    ConfigurationContractRegistry,
    UnknownConfigurationContractError,
    build_builtin_configuration_contract_registry,
    build_configuration_contract_registry,
    validate_builtin_policy_contract_consistency,
)
from intergrax.runtime.policy.contribution import (
    GovernancePolicyContribution,
    build_composed_configuration_contract_registry,
    build_composed_policy_catalog,
    load_policy_definition_plugin_report,
)
from intergrax.runtime.policy.builtin_catalog import (
    TOOL_INVOCATION_CONTROL_POLICY_ID,
    TOOL_INVOCATION_CONTROL_VERSION,
    build_builtin_policy_catalog,
    build_policy_catalog,
)
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.plugin_loader import (
    PolicyRuleLoadPolicy,
    load_policy_rule_plugin_report,
)
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import PolicyRuleAction

pytestmark = pytest.mark.unit

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
    rule_id = _HANDLER_ID

    def evaluate(
        self,
        rule: object,
        *,
        context: PolicyEvaluationContext,
    ) -> object:
        return PolicyRuleAction.ALLOW


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


_PLUGIN_BINDING = ConfigurationContractBinding.from_pydantic_model(
    _CONFIG_CONTRACT_ID,
    _PluginTestConfig,
)


def _governance_contribution(
    definition: PolicyDefinition | None = None,
    *,
    package_name: str = _PACKAGE_NAME,
    configuration_contract_binding: ConfigurationContractBinding | None = _PLUGIN_BINDING,
) -> GovernancePolicyContribution:
    return GovernancePolicyContribution(
        definition=definition or _plugin_policy_definition(),
        package_identity=DistributionPackageIdentity(
            name=package_name,
            version=_PACKAGE_VERSION,
        ),
        configuration_contract_binding=configuration_contract_binding,
    )


_CONTRIBUTION_EP = _governance_contribution()
_DEFINITION_ONLY = _plugin_policy_definition()


def _builtin_override_contribution() -> GovernancePolicyContribution:
    return _governance_contribution(
        _plugin_policy_definition(
            configuration_contract_id=TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
        ),
        configuration_contract_binding=ConfigurationContractBinding.from_pydantic_model(
            TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
            ToolInvocationControlConfig,
        ),
    )


_INVALID_BINDING_CONTRIBUTION = _governance_contribution(
    configuration_contract_binding=ConfigurationContractBinding.from_pydantic_model(
        "acme.mismatched.v1",
        _PluginTestConfig,
    ),
)


_MIXED_INVALID_BINDING = (_CONTRIBUTION_EP, _INVALID_BINDING_CONTRIBUTION)


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _rule_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", EP_POLICY_RULES, distribution=distribution)


def _definition_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        EP_POLICY_DEFINITIONS,
        distribution=distribution,
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
                ref="tests/unit/runtime/policy/test_configuration_contract_registry.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )


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


def _lookup_for(qualification: object | None) -> object:
    return lambda spec: qualification


def _production_load_policy(qualification: object | None) -> PolicyRuleLoadPolicy:
    return PolicyRuleLoadPolicy(
        require_production_admission=True,
        package_qualification_lookup=_lookup_for(qualification),
        platform_version=_PLATFORM_VERSION,
    )


def _load_admitted_handlers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    qualification: object | None = None,
    package_name: str = _PACKAGE_NAME,
) -> tuple[PolicyRuleRegistry, tuple[object, ...]]:
    _install_eps(
        monkeypatch,
        [_rule_ep("alpha", "_AlphaHandler", distribution=package_name)],
    )
    _mock_installed_distribution(monkeypatch, package_name=package_name)
    compatibility = _compatible_platform()
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    registry = PolicyRuleRegistry()
    outcome = load_policy_rule_plugin_report(
        registry,
        policy=_production_load_policy(qualification),
    )
    return registry, outcome.handler_provenance


def test_builtin_resolve_returns_exact_binding() -> None:
    registry = build_builtin_configuration_contract_registry()
    binding = registry.resolve(TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID)
    assert binding.contract_id == TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID


def test_valid_config_returns_typed_object() -> None:
    registry = build_builtin_configuration_contract_registry()
    result = registry.validate(
        TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
        {"tool_id": "foo", "action": PolicyRuleAction.DENY},
    )
    assert isinstance(result, ToolInvocationControlConfig)
    assert result.tool_id == "foo"
    assert result.action is PolicyRuleAction.DENY


def test_invalid_field_raises_validation_error() -> None:
    registry = build_builtin_configuration_contract_registry()
    with pytest.raises(ValidationError):
        registry.validate(
            TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
            {"tool_id": "foo", "action": PolicyRuleAction.DENY, "extra": True},
        )


def test_empty_tool_id_raises_validation_error() -> None:
    registry = build_builtin_configuration_contract_registry()
    with pytest.raises(ValidationError):
        registry.validate(
            TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
            {"tool_id": "   ", "action": PolicyRuleAction.DENY},
        )


def test_invalid_action_raises_validation_error() -> None:
    registry = build_builtin_configuration_contract_registry()
    with pytest.raises(ValidationError):
        registry.validate(
            TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
            {"tool_id": "foo", "action": "not-a-real-action"},
        )


def test_unknown_contract_id_raises() -> None:
    registry = build_builtin_configuration_contract_registry()
    with pytest.raises(UnknownConfigurationContractError) as exc:
        registry.resolve("missing.contract.v1")
    assert exc.value.contract_id == "missing.contract.v1"


def test_duplicate_contract_id_raises_conflict() -> None:
    duplicate = ConfigurationContractBinding.from_pydantic_model(
        TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
        ToolInvocationControlConfig,
    )
    with pytest.raises(ConfigurationContractConflictError) as exc:
        build_configuration_contract_registry(plugin_bindings=(duplicate,))
    assert exc.value.contract_id == TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID


def test_builtin_policy_contract_consistency() -> None:
    catalog = build_builtin_policy_catalog()
    registry = build_builtin_configuration_contract_registry()
    validate_builtin_policy_contract_consistency(catalog, registry)
    definition = catalog.resolve(
        policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
        version=TOOL_INVOCATION_CONTROL_VERSION,
    )
    binding = registry.resolve(definition.configuration_contract_id)
    validated = binding.validate({"tool_id": "demo", "action": PolicyRuleAction.ALLOW})
    assert isinstance(validated, ToolInvocationControlConfig)


def test_qualified_plugin_binding_in_catalog_and_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=qualification,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    catalog, outcome = build_composed_policy_catalog(
        registry,
        handler_provenance,
        policy=_production_load_policy(qualification),
    )
    config_registry = build_composed_configuration_contract_registry(outcome.contributions)
    resolved = catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)
    binding = config_registry.resolve(resolved.configuration_contract_id)
    validated = binding.validate({"enabled": True})
    assert isinstance(validated, _PluginTestConfig)
    assert validated.enabled is True


def test_plugin_definition_missing_binding_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=qualification,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_DEFINITION_ONLY", distribution=_PACKAGE_NAME),
        ],
    )
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(qualification),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.POLICY_CONFIGURATION_CONTRACT_BINDING_MISSING
    )


def test_plugin_definition_cannot_use_foreign_package_binding(
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
                ref="tests/unit/runtime/policy/test_configuration_contract_registry.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=alpha_qualification,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep(
                "beta-policy",
                "_CONTRIBUTION_EP",
                distribution=_OTHER_PACKAGE_NAME,
            ),
        ],
    )
    _mock_installed_distribution(
        monkeypatch,
        extra_packages={_OTHER_PACKAGE_NAME: _PACKAGE_VERSION},
    )
    load_policy = PolicyRuleLoadPolicy(
        require_production_admission=True,
        package_qualification_lookup=lambda spec: (
            alpha_qualification
            if spec.distribution == _PACKAGE_NAME
            else beta_qualification
            if spec.distribution == _OTHER_PACKAGE_NAME
            else None
        ),
        platform_version=_PLATFORM_VERSION,
    )
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=load_policy,
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.POLICY_HANDLER_PROVENANCE_MISMATCH
    )


def test_plugin_builtin_contract_id_override_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=qualification,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep(
                "builtin-override",
                "_builtin_override_contribution",
                distribution=_PACKAGE_NAME,
            ),
        ],
    )
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(qualification),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.CONFIGURATION_CONTRACT_BUILTIN_RESERVED
    )


def test_invalid_binding_rejects_entire_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=qualification,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep(
                "alpha-policy",
                "_MIXED_INVALID_BINDING",
                distribution=_PACKAGE_NAME,
            ),
        ],
    )
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(qualification),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.POLICY_CONFIGURATION_CONTRACT_ID_MISMATCH
    )
    assert outcome.report.accepted == ()


def test_canonical_runtime_bundle_policy_to_typed_config(
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
    compatibility = _compatible_platform()
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )

    bundle = build_runtime_policy_bundle(
        policy_rules=PolicyRulesProfile(inline_rules=[]),
        discover_entry_points=True,
        package_qualification_lookup=_lookup_for(qualification),
        execution_mode=None,
    )
    assert bundle.declarative_policy_runtime is not None
    definition = bundle.policy_catalog.resolve(
        policy_id=_POLICY_ID,
        version=_POLICY_VERSION,
    )
    binding = bundle.configuration_contract_registry.resolve(
        definition.configuration_contract_id,
    )
    validated = binding.validate({"enabled": False})
    assert isinstance(validated, _PluginTestConfig)
    assert validated.enabled is False
    assert bundle.declarative_policy_runtime.registry.resolve(_HANDLER_ID) is not None


def test_policy_rules_none_uses_builtin_registry_only() -> None:
    bundle = build_runtime_policy_bundle(policy_rules=None, discover_entry_points=True)
    registry = bundle.configuration_contract_registry
    assert len(registry.bindings()) == 1
    assert registry.resolve(TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID) is not None
    validate_builtin_policy_contract_consistency(bundle.policy_catalog, registry)


def test_discover_entry_points_false_uses_builtin_registry_only() -> None:
    bundle = build_runtime_policy_bundle(
        policy_rules=PolicyRulesProfile(inline_rules=[]),
        discover_entry_points=False,
    )
    registry = bundle.configuration_contract_registry
    assert len(registry.bindings()) == 1
    catalog = build_policy_catalog()
    assert bundle.policy_catalog.definitions() == catalog.definitions()


def test_empty_registry_constructs() -> None:
    registry = ConfigurationContractRegistry()
    assert registry.bindings() == ()
