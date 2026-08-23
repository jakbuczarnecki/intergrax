# © Artur Czarnecki. All rights reserved.

"""PLUGIN-SEC-ADOPTION-1: Tier-3 canonical security plugin bootstrap ownership."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.applications._shared import environment_wiring as environment_wiring_module
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.security_assembly_resolver import SecurityAssemblyError
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_SECURITY,
)
from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.plugin_env import INTERGRAX_DISCOVER_PLUGINS_ENV
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import EP_SECURITY_DEFENSES, reset_entry_point_spec_cache_for_tests
from intergrax.core.security_bootstrap import bootstrap_security_providers
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import SecurityFailMode, SecurityInspectionResult
from intergrax.runtime.security.defense_registry import (
    get_security_defense_plugin,
    reset_security_defense_registry_for_tests,
)
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GROUP = "intergrax.security_defenses"


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


class _AlphaDefense:
    plugin_id = "lab.alpha"
    version = "1.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _GammaDefense:
    plugin_id = "lab.alpha"
    version = "1.0.0"
    hook_points = frozenset()
    priority = 0
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: object) -> SecurityInspectionResult:
        return SecurityInspectionResult(plugin_id=self.plugin_id, hook_point=str(point))


class _NotADefense:
    pass


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


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", _GROUP)


def _strict_env(profile_id: str) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    return env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
        },
    )


def test_bootstrap_catalogs_does_not_invoke_security_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = bootstrap_security_providers

    def _spy(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(
        "intergrax.core.security_bootstrap.bootstrap_security_providers",
        _spy,
    )
    bootstrap_catalogs(register_shipped=False, discover_entry_points=False)
    bootstrap_catalogs(register_shipped=False, discover_entry_points=False)
    assert calls == []


@pytest.mark.no_ci
def test_wire_application_environment_bootstraps_security_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = environment_wiring_module.bootstrap_security_providers

    def _count(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(environment_wiring_module, "bootstrap_security_providers", _count)
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaDefense")])

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.adoption.once")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    assert len(calls) == 1
    security_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SECURITY)
    assert security_report is not None
    assert security_report.group == EP_SECURITY_DEFENSES
    assert [item.name for item in security_report.accepted] == ["alpha"]
    assert security_report.critical_bootstrap_acceptable is True
    assert not any(
        item.reason_code is PluginAdmissionReasonCode.ALREADY_REGISTERED
        for item in security_report.rejected
    )
    assert get_security_defense_plugin("lab.alpha") is not None


@pytest.mark.no_ci
def test_wire_application_environment_security_evidence_matches_canonical_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []
    original = environment_wiring_module._bootstrap_application_security_providers

    def _capture() -> object:
        result = original()
        captured.append(result.load_report)
        return result

    monkeypatch.setattr(
        environment_wiring_module,
        "_bootstrap_application_security_providers",
        _capture,
    )
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.adoption.evidence")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    assert captured
    assert wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SECURITY) is captured[0]


@pytest.mark.no_ci
def test_strict_wire_application_environment_fails_on_invalid_security_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(
        monkeypatch,
        [
            _ep("alpha", "_AlphaDefense"),
            _ep("nope", "_NotADefense"),
        ],
    )
    settings = LabApplicationSettings.from_env()
    env = _strict_env("sec.adoption.strict-invalid")

    with pytest.raises(SecurityAssemblyError, match="invalid_target_type"):
        wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)


@pytest.mark.no_ci
def test_strict_wire_application_environment_fails_on_plugin_id_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(
        monkeypatch,
        [
            _ep("foo", "_AlphaDefense"),
            _ep("bar", "_GammaDefense"),
        ],
    )
    settings = LabApplicationSettings.from_env()
    env = _strict_env("sec.adoption.strict-collision")

    with pytest.raises(SecurityAssemblyError, match="plugin_id_collision"):
        wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)


@pytest.mark.no_ci
def test_strict_wire_application_environment_allows_valid_security_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaDefense")])
    settings = LabApplicationSettings.from_env()
    env = _strict_env("sec.adoption.strict-ok")

    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    security_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SECURITY)
    assert security_report is not None
    assert security_report.critical_bootstrap_acceptable is True
    assert get_security_defense_plugin("lab.alpha") is not None


@pytest.mark.no_ci
def test_wire_application_environment_discovery_disabled_security_evidence_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(INTERGRAX_DISCOVER_PLUGINS_ENV, raising=False)
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaDefense")])
    settings = LabApplicationSettings.from_env()
    env = _strict_env("sec.adoption.discovery-off")

    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    security_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SECURITY)
    assert security_report is not None
    assert security_report.registered_count == 0
    assert security_report.accepted == ()
    assert security_report.critical_bootstrap_acceptable is True
    assert get_security_defense_plugin("harness.strict_injection") is not None
