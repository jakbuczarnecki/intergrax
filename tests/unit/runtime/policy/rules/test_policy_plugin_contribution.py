# © Artur Czarnecki. All rights reserved.

"""Typed plugin PolicyDefinition contribution tests (G4B-2)."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.core.distribution import DistributionPackageIdentity, PlatformCompatibility, check_platform_compatibility
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_POLICY_DEFINITIONS,
    EP_POLICY_RULES,
    EntryPointSpec,
    load_entry_point_value,
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
from intergrax.contracts.tool_invocation_control_policy import (
    TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
)
from intergrax.runtime.policy.builtin_catalog import (
    TOOL_INVOCATION_CONTROL_POLICY_ID,
    TOOL_INVOCATION_CONTROL_VERSION,
    build_policy_catalog,
)
from intergrax.runtime.policy.catalog import (
    PolicyDefinitionConflictError,
    UnknownPolicyDefinitionError,
    UnsupportedPolicyDefinitionVersionError,
)
from intergrax.runtime.policy.contribution import (
    GovernancePolicyContribution,
    build_composed_policy_catalog,
    load_policy_definition_plugin_report,
)
from intergrax.runtime.policy.configuration_contract import ConfigurationContractBinding
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.plugin_loader import (
    PolicyRuleLoadPolicy,
    load_policy_rule_plugin_report,
)
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import PolicyRuleAction

pytestmark = pytest.mark.unit

_RULES_GROUP = EP_POLICY_RULES
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


def _builtin_spoof() -> PolicyDefinition:
    return _plugin_policy_definition(source=PolicyDefinitionSource.BUILT_IN)


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


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _rule_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        _RULES_GROUP,
        distribution=distribution,
    )


def _definition_ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        _DEFINITIONS_GROUP,
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
                ref="tests/unit/runtime/policy/rules/test_policy_plugin_contribution.py",
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


def _lookup_for_packages(qualifications: dict[str, object | None]) -> object:
    def lookup(spec: EntryPointSpec) -> object | None:
        if spec.distribution is None:
            return None
        return qualifications.get(spec.distribution)

    return lookup


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
        policy=PolicyRuleLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=_lookup_for(qualification),
            platform_version=_PLATFORM_VERSION,
        ),
    )
    return registry, outcome.handler_provenance


def _production_load_policy(qualification: object | None) -> PolicyRuleLoadPolicy:
    return PolicyRuleLoadPolicy(
        require_production_admission=True,
        package_qualification_lookup=_lookup_for(qualification),
        platform_version=_PLATFORM_VERSION,
    )


def test_admitted_plugin_contribution_resolves_in_catalog(
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
    resolved = catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)
    assert resolved.source is PolicyDefinitionSource.PLUGIN
    assert resolved.handler_id == _HANDLER_ID
    assert outcome.report.registered_count == 1


def test_production_rejected_package_has_no_contribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=None,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(None),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED
    )


def test_missing_handler_binding_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    registry = PolicyRuleRegistry()
    handler_provenance: tuple[object, ...] = ()
    _install_eps(
        monkeypatch,
        [_definition_ep("alpha-policy", "_CONTRIBUTION", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(qualification),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.POLICY_HANDLER_BINDING_MISSING
    )


def test_handler_provenance_package_mismatch_rejected(
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
                ref="tests/unit/runtime/policy/rules/test_policy_plugin_contribution.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=alpha_qualification,
        package_name=_PACKAGE_NAME,
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
        package_name=_PACKAGE_NAME,
        package_version=_PACKAGE_VERSION,
        extra_packages={_OTHER_PACKAGE_NAME: _PACKAGE_VERSION},
    )
    load_policy = PolicyRuleLoadPolicy(
        require_production_admission=True,
        package_qualification_lookup=_lookup_for_packages(
            {
                _PACKAGE_NAME: alpha_qualification,
                _OTHER_PACKAGE_NAME: beta_qualification,
            }
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


def test_builtin_source_spoof_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=qualification,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_builtin_spoof", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(qualification),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.INVALID_POLICY_CONTRIBUTION_SOURCE
    )


def test_plugin_builtin_exact_identity_conflict(
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
                "conflict",
                "_conflicting_contribution",
                distribution=_PACKAGE_NAME,
            ),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    with pytest.raises(PolicyDefinitionConflictError) as exc:
        build_composed_policy_catalog(
            registry,
            handler_provenance,
            policy=_production_load_policy(qualification),
        )
    assert exc.value.policy_id == TOOL_INVOCATION_CONTROL_POLICY_ID
    assert exc.value.version == TOOL_INVOCATION_CONTROL_VERSION


def test_two_plugin_contributions_same_identity_conflict(
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
            _definition_ep("first", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
            _definition_ep("second", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    with pytest.raises(PolicyDefinitionConflictError) as exc:
        build_composed_policy_catalog(
            registry,
            handler_provenance,
            policy=_production_load_policy(qualification),
        )
    assert exc.value.policy_id == _POLICY_ID
    assert exc.value.version == _POLICY_VERSION


def test_package_version_differs_from_policy_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_version = "3.4.1"
    qualification = build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=_PACKAGE_NAME,
            package_version=package_version,
            domain="policy",
            entry_point_group=EP_POLICY_RULES,
            entry_point_name="alpha",
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            compatibility_evidence(_compatible_platform()),
            QualificationEvidence(
                kind=PluginQualificationEvidenceKind.DOMAIN_QUALIFICATION,
                code="policy.tests.passed",
                ref="tests/unit/runtime/policy/rules/test_policy_plugin_contribution.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    _mock_installed_distribution(
        monkeypatch,
        package_name=_PACKAGE_NAME,
        package_version=package_version,
    )
    compatibility = _compatible_platform()
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    registry = PolicyRuleRegistry()
    handler_outcome = load_policy_rule_plugin_report(
        registry,
        policy=_production_load_policy(qualification),
    )
    catalog, outcome = build_composed_policy_catalog(
        registry,
        handler_outcome.handler_provenance,
        policy=_production_load_policy(qualification),
    )
    resolved = catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)
    assert resolved.version == _POLICY_VERSION
    assert outcome.contributions[0].package_identity == DistributionPackageIdentity(
        name=_PACKAGE_NAME,
        version=package_version,
    )
    assert outcome.contributions[0].package_identity.version != resolved.version


def test_unknown_policy_uses_existing_error() -> None:
    catalog = build_policy_catalog()
    with pytest.raises(UnknownPolicyDefinitionError):
        catalog.resolve(policy_id="missing-policy", version="1")


def test_unsupported_version_uses_existing_error() -> None:
    catalog = build_policy_catalog()
    with pytest.raises(UnsupportedPolicyDefinitionVersionError):
        catalog.resolve(
            policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
            version="999",
        )


def test_end_to_end_handler_admission_to_catalog_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[str] = []

    def _tracking_load(value: str) -> object:
        load_calls.append(value)
        return load_entry_point_value(value)

    monkeypatch.setattr(
        "intergrax.core.plugins.discovery.load_entry_point_value",
        _tracking_load,
    )
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
    registry = PolicyRuleRegistry()
    handler_outcome = load_policy_rule_plugin_report(
        registry,
        policy=_production_load_policy(qualification),
    )
    assert registry.resolve(_HANDLER_ID) is not None
    assert handler_outcome.handler_provenance[0].package_identity == DistributionPackageIdentity(
        name=_PACKAGE_NAME,
        version=_PACKAGE_VERSION,
    )

    catalog, definition_outcome = build_composed_policy_catalog(
        registry,
        handler_outcome.handler_provenance,
        policy=_production_load_policy(qualification),
    )
    contribution = GovernancePolicyContribution(
        definition=_CONTRIBUTION,
        package_identity=DistributionPackageIdentity(
            name=_PACKAGE_NAME,
            version=_PACKAGE_VERSION,
        ),
        configuration_contract_binding=_PLUGIN_BINDING,
    )
    assert definition_outcome.contributions[0] == contribution
    resolved = catalog.resolve(policy_id=_POLICY_ID, version=_POLICY_VERSION)
    assert resolved.handler_id == _HANDLER_ID
    assert resolved.source is PolicyDefinitionSource.PLUGIN
    assert all(
        call.endswith((":_AlphaHandler", ":_CONTRIBUTION_EP"))
        for call in load_calls
    )


def test_production_rejected_definition_not_imported_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = _production_package_qualification(compatibility=_compatible_platform())
    registry, handler_provenance = _load_admitted_handlers(
        monkeypatch,
        qualification=qualification,
    )
    definition_load_calls: list[str] = []
    original_load = load_entry_point_value

    def _tracking_load(value: str) -> object:
        if value.endswith((":_CONTRIBUTION", ":_CONTRIBUTION_EP")):
            definition_load_calls.append(value)
        return original_load(value)

    monkeypatch.setattr(
        "intergrax.core.plugins.discovery.load_entry_point_value",
        _tracking_load,
    )
    _install_eps(
        monkeypatch,
        [
            _rule_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _definition_ep("alpha-policy", "_CONTRIBUTION_EP", distribution=_PACKAGE_NAME),
        ],
    )
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(None),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED
    )
    assert definition_load_calls == []


_MIXED_CONTRIBUTIONS = (_CONTRIBUTION_EP, _builtin_spoof())


def test_multi_value_entry_point_rejects_all_when_any_invalid(
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
                "_MIXED_CONTRIBUTIONS",
                distribution=_PACKAGE_NAME,
            ),
        ],
    )
    _mock_installed_distribution(monkeypatch)
    outcome = load_policy_definition_plugin_report(
        registry,
        handler_provenance,
        policy=_production_load_policy(qualification),
    )
    assert outcome.contributions == ()
    assert outcome.report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.INVALID_POLICY_CONTRIBUTION_SOURCE
    )
    assert outcome.report.accepted == ()
