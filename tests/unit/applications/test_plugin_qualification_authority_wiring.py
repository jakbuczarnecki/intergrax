# © Artur Czarnecki. All rights reserved.

"""G4B-1B: canonical host qualification authority wiring."""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.core.distribution import DistributionPackageIdentity, PlatformCompatibility, check_platform_compatibility
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import EP_POLICY_RULES, reset_entry_point_spec_cache_for_tests
from intergrax.core.plugins.platform_qualification import (
    PlatformPluginPackageQualificationBundle,
    PluginQualificationEvidenceKind,
    PluginQualificationLevel,
    build_external_package_subject,
    build_qualification_result,
    compatibility_evidence,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.schema import PolicyRuleAction
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

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


def _strict_env() -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="g4b1b.strict")
    return env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
            "policy_rules": _inline_profile(),
        },
    )


def _balanced_env() -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="g4b1b.balanced")
    return env.model_copy(update={"policy_rules": _inline_profile()})


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
                ref="tests/unit/applications/test_plugin_qualification_authority_wiring.py",
            ),
        ),
        reason="external policy plugin production-qualified",
    )


def _qualification_bundle(
    qualification: object | None,
    *,
    version: str = _PACKAGE_VERSION,
) -> PlatformPluginPackageQualificationBundle | None:
    if qualification is None:
        return None
    identity = DistributionPackageIdentity(name=_PACKAGE_NAME, version=version)
    return PlatformPluginPackageQualificationBundle([(identity, qualification)])


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


def _wire_policy(
    env: ApplicationEnvironmentProfile,
    *,
    qualifications: PlatformPluginPackageQualificationBundle | None,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    return wire_policy_bundle(env, package_qualifications=qualifications)


def test_strict_canonical_wiring_without_bundle_rejects_external_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)])
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: _compatible_platform(),
    )
    bundle = _wire_policy(_strict_env(), qualifications=None, monkeypatch=monkeypatch)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is None
    assert runtime.load_report.rejected[0].reason_code is (
        PluginAdmissionReasonCode.PRODUCTION_ADMISSION_DENIED
    )


def test_strict_canonical_wiring_unknown_package_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    other_name = "other-package"
    qualification = build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=other_name,
            package_version=_PACKAGE_VERSION,
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="qualified other package",
    )
    other_identity = DistributionPackageIdentity(name=other_name, version=_PACKAGE_VERSION)
    qualifications = PlatformPluginPackageQualificationBundle(
        [(other_identity, qualification)],
    )
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)])
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    bundle = _wire_policy(_strict_env(), qualifications=qualifications, monkeypatch=monkeypatch)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is None


def test_strict_canonical_wiring_wrong_version_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    wrong_version = "9.9.9"
    qualification = build_qualification_result(
        subject=build_external_package_subject(
            level=PluginQualificationLevel.PACKAGE,
            package_name=_PACKAGE_NAME,
            package_version=wrong_version,
        ),
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(compatibility_evidence(compatibility),),
        reason="qualified wrong version",
    )
    qualifications = PlatformPluginPackageQualificationBundle(
        [
            (
                DistributionPackageIdentity(name=_PACKAGE_NAME, version=wrong_version),
                qualification,
            )
        ],
    )
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)])
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    bundle = _wire_policy(_strict_env(), qualifications=qualifications, monkeypatch=monkeypatch)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is None


def test_strict_canonical_wiring_not_production_qualified_not_registered(
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
    qualifications = _qualification_bundle(qualification)
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)])
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    bundle = _wire_policy(_strict_env(), qualifications=qualifications, monkeypatch=monkeypatch)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is None


def test_strict_canonical_wiring_production_qualified_registers_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    qualification = _production_package_qualification(compatibility=compatibility)
    qualifications = _qualification_bundle(qualification)
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)])
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    bundle = _wire_policy(_strict_env(), qualifications=qualifications, monkeypatch=monkeypatch)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is not None
    assert runtime.load_report.registered_count == 1


def test_balanced_canonical_wiring_preserves_external_plugin_without_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)])
    bundle = _wire_policy(_balanced_env(), qualifications=None, monkeypatch=monkeypatch)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is not None


@pytest.mark.no_ci
def test_environment_wiring_threads_qualification_bundle_to_policy_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = _compatible_platform()
    qualification = _production_package_qualification(compatibility=compatibility)
    qualifications = _qualification_bundle(qualification)
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler", distribution=_PACKAGE_NAME)])
    _mock_installed_distribution(monkeypatch)
    monkeypatch.setattr(
        "intergrax.core.plugins.platform_qualification.resolve_installed_distribution_platform_compatibility",
        lambda *_args, **_kwargs: compatibility,
    )
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    settings = LabApplicationSettings.from_env()
    env = _strict_env()
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        conformance_check=False,
        platform_plugin_package_qualifications=qualifications,
    )
    runtime = wiring.policy_bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.registry.resolve(_HANDLER_ID) is not None
