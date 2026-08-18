# © Artur Czarnecki. All rights reserved.

"""Production admission hardening for external policy rule plugins (G4B-1)."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.environment_profile import PolicyRulesProfile
from intergrax.core.distribution import PlatformCompatibility, check_platform_compatibility
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_POLICY_RULES,
    load_entry_point_value,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.core.plugins.platform_qualification import (
    PluginQualificationEvidenceKind,
    PluginQualificationLevel,
    build_external_package_subject,
    build_qualification_result,
    compatibility_evidence,
    resolve_installed_distribution_platform_compatibility,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.plugin_loader import (
    PolicyRuleLoadPolicy,
    load_policy_rule_plugin_report,
)
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import PolicyRuleAction

pytestmark = pytest.mark.unit

_GROUP = EP_POLICY_RULES
_PACKAGE_NAME = "alpha-policy-plugin"
_PACKAGE_VERSION = "1.0.0"
_HANDLER_ID = "alpha-rule"
_PLATFORM_VERSION = "0.1.0"


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


class _ShippedIdHandler:
    rule_id = "deny_tool"

    def evaluate(
        self,
        rule: object,
        *,
        context: PolicyEvaluationContext,
    ) -> object:
        return PolicyRuleAction.ALLOW


class _BetaHandler:
    rule_id = "beta-rule"

    def evaluate(
        self,
        rule: object,
        *,
        context: PolicyEvaluationContext,
    ) -> object:
        return PolicyRuleAction.ALLOW


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _ep(name: str, attr: str, *, distribution: str | None = None) -> _EntryPoint:
    return _EntryPoint(
        name,
        f"{__name__}:{attr}",
        _GROUP,
        distribution=distribution,
    )


def _inline_profile() -> PolicyRulesProfile:
    return PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "wiring.blocked",
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": "blocked",
                "action": "deny",
            }
        ],
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
                ref="tests/unit/runtime/policy/rules/test_policy_plugin_production_admission.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )


def _mock_installed_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    dist = MagicMock()
    dist.version = _PACKAGE_VERSION
    dist.files = None
    intergrax_dist = MagicMock()
    intergrax_dist.version = _PLATFORM_VERSION

    def _distribution(name: str) -> MagicMock:
        if name == _PACKAGE_NAME:
            return dist
        if name in ("intergrax", "Intergrax-ai"):
            return intergrax_dist
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", _distribution)


def test_production_entry_point_distribution_none_rejected_before_import(
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
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=None)],
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=None,
            platform_version=_PLATFORM_VERSION,
        ),
    ).report
    assert load_calls == []
    assert registry.resolve(_HANDLER_ID) is None
    assert report.registered_count == 0
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED
    assert report.rejected[0].spec.name == "alpha"
    assert report.rejected[0].spec.distribution is None


def test_production_external_plugin_missing_qualification_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=None,
            platform_version=_PLATFORM_VERSION,
        ),
    ).report
    assert registry.resolve(_HANDLER_ID) is None
    assert report.registered_count == 0
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED
    assert report.rejected[0].spec.name == "alpha"


def _lookup_for(qualification: object | None) -> object:
    return lambda spec: qualification


def test_production_external_plugin_not_production_qualified_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    qualification = build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=_PACKAGE_NAME,
            package_version=_PACKAGE_VERSION,
        ),
        status=QualificationStatus.QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="qualified only",
    )
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=_lookup_for(qualification),
            platform_version=_PLATFORM_VERSION,
        ),
    ).report
    assert registry.resolve(_HANDLER_ID) is None
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED


def test_production_external_plugin_incompatible_platform_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incompatible = check_platform_compatibility(
        PlatformCompatibility(intergrax_version=">=2,<3"),
        _PLATFORM_VERSION,
    )
    qualification = _production_package_qualification(compatibility=incompatible)
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: incompatible,
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(
            require_production_admission=True,
            package_qualification_lookup=_lookup_for(qualification),
            platform_version=_PLATFORM_VERSION,
        ),
    ).report
    assert registry.resolve(_HANDLER_ID) is None
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED


def test_production_external_plugin_valid_qualification_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    qualification = _production_package_qualification(compatibility=compatibility)
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
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
    assert registry.resolve(_HANDLER_ID) is not None
    assert outcome.report.registered_count == 1
    assert outcome.report.accepted[0].name == "alpha"
    assert outcome.handler_provenance[0].rule_id == _HANDLER_ID


def test_non_production_path_preserves_plugin_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)],
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(require_production_admission=False),
    ).report
    assert registry.resolve(_HANDLER_ID) is not None
    assert report.registered_count == 1


def test_strict_wiring_blocks_external_plugin_without_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)],
    )
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_inline_profile(),
        execution_mode=ExecutionMode.STRICT,
        discover_entry_points=True,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is None
    assert runtime.load_report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED
    )


def test_balanced_wiring_preserves_external_plugin_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)],
    )
    bundle = build_runtime_policy_bundle(
        policy_rules=_inline_profile(),
        execution_mode=ExecutionMode.BALANCED,
        discover_entry_points=True,
    )
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is not None


def test_shipped_handler_protection_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [_ep("patch", "_ShippedIdHandler", distribution=_PACKAGE_NAME)],
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(require_production_admission=False),
    ).report
    assert registry.resolve("deny_tool") is not None
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.SHIPPED_ID_COLLISION


def test_allowlist_behavior_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME),
            _ep("beta", "_BetaHandler", distribution=_PACKAGE_NAME),
        ],
    )
    registry = PolicyRuleRegistry()
    report = load_policy_rule_plugin_report(
        registry,
        policy=PolicyRuleLoadPolicy(
            require_production_admission=False,
            allowed_handler_ids=frozenset({"beta-rule"}),
        ),
    ).report
    assert registry.resolve(_HANDLER_ID) is None
    assert registry.resolve("beta-rule") is not None
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.NOT_IN_ALLOWLIST
