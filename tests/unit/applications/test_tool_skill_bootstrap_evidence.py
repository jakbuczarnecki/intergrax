# © Artur Czarnecki. All rights reserved.

"""Stage 10: Tool and Skill bootstrap evidence in Tier-3 application wiring."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.applications._shared import environment_wiring as environment_wiring_module
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.skill_wiring import SkillCatalogBootstrapError
from intergrax.applications._shared.tool_wiring import ToolCatalogBootstrapError
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_CONTEXT,
    PLATFORM_PLUGIN_DOMAIN_MEMORY,
    PLATFORM_PLUGIN_DOMAIN_SECURITY,
    PLATFORM_PLUGIN_DOMAIN_SKILLS,
    PLATFORM_PLUGIN_DOMAIN_TOOLS,
)
from intergrax.core.catalog_bootstrap import bootstrap_catalogs, reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.plugin_env import INTERGRAX_DISCOVER_PLUGINS_ENV
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import (
    EP_SKILLS,
    EP_TOOLS,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.core.manifest import ToolBundleManifest
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import catalog_snapshot as tool_catalog_snapshot
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.catalog import ToolBundleStatus
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.catalog import catalog_snapshot as skill_catalog_snapshot
from intergrax.skills.registry.runtime import SkillRegistry
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter
from intergrax.skills.registry.profile import SkillProfile

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gate,
    pytest.mark.usefixtures("catalog_fixture_installed"),
]

_TOOL_GROUP = EP_TOOLS
_SKILL_GROUP = EP_SKILLS
_COLLISION_BUNDLE_ID = "stage10_collision"
_SKILL_COLLISION_BUNDLE_ID = "stage10_skill_collision"


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


class _NotATool:
    pass


class _NotASkill:
    pass


class _CollisionAlphaTool:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id=_COLLISION_BUNDLE_ID,
            tool_ids=("stage10.collision.alpha",),
            status=ToolBundleStatus.BETA,
            description="collision alpha",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        pass


class _CollisionBetaTool:
    @classmethod
    def tool_bundle_manifest(cls) -> ToolBundleManifest:
        return ToolBundleManifest(
            bundle_id=_COLLISION_BUNDLE_ID,
            tool_ids=("stage10.collision.beta",),
            status=ToolBundleStatus.BETA,
            description="collision beta",
        )

    @classmethod
    def register_tools(cls, registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        pass


class _CollisionAlphaSkill:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id=_SKILL_COLLISION_BUNDLE_ID,
            skill_ids=("stage10.skill.collision.alpha",),
            status=SkillBundleStatus.BETA,
            description="collision alpha",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return ()

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        from intergrax.skills.core.contracts import SkillManifest

        registry.register(
            SkillManifest(
                skill_id="stage10.skill.collision.alpha",
                description="collision alpha",
            ),
        )


class _CollisionBetaSkill:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id=_SKILL_COLLISION_BUNDLE_ID,
            skill_ids=("stage10.skill.collision.beta",),
            status=SkillBundleStatus.BETA,
            description="collision beta",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return ()

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        from intergrax.skills.core.contracts import SkillManifest

        registry.register(
            SkillManifest(
                skill_id="stage10.skill.collision.beta",
                description="collision beta",
            ),
        )


@pytest.fixture(autouse=True)
def _stub_environment_llm_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        environment_wiring_module,
        "resolve_environment_llm_adapter",
        lambda _env: FakeLLMAdapter(),
    )


@pytest.fixture(autouse=True)
def _reset_catalog_state() -> None:
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
    yield
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _tool_ep(name: str, value: str) -> _EntryPoint:
    resolved = value if ":" in value else f"{__name__}:{value}"
    return _EntryPoint(name, resolved, _TOOL_GROUP)


def _skill_ep(name: str, value: str) -> _EntryPoint:
    resolved = value if ":" in value else f"{__name__}:{value}"
    return _EntryPoint(name, resolved, _SKILL_GROUP)


def _strict_env(profile_id: str) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    return env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
        },
    )


def test_wire_application_environment_exposes_tools_and_skills_domain_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(
        monkeypatch,
        [
            _tool_ep(
                "fixture_ep",
                "intergrax_catalog_fixture.tool:FixtureEchoToolPlugin",
            ),
            _skill_ep(
                "fixture_ep",
                "intergrax_catalog_fixture.skill:FixturePackSkillPlugin",
            ),
        ],
    )
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.reports")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    tools_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_TOOLS)
    skills_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SKILLS)
    assert tools_report is not None
    assert skills_report is not None
    assert tools_report.group == EP_TOOLS
    assert skills_report.group == EP_SKILLS
    assert [item.name for item in tools_report.accepted] == ["fixture_ep"]
    assert [item.name for item in skills_report.accepted] == ["fixture_ep"]
    assert tools_report.registered_count == 1
    assert skills_report.registered_count == 1
    assert tools_report.critical_bootstrap_acceptable is True
    assert skills_report.critical_bootstrap_acceptable is True


def test_tool_skill_reports_match_same_catalog_bootstrap_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []
    original = bootstrap_catalogs

    def _capture(**kwargs: object) -> object:
        result = original(**kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(environment_wiring_module, "bootstrap_catalogs", _capture)
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.same-pass")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    assert len(captured) == 1
    bootstrap_result = captured[0]
    assert (
        wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_TOOLS)
        is bootstrap_result.tool_plugin_load_report
    )
    assert (
        wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SKILLS)
        is bootstrap_result.skill_plugin_load_report
    )


def test_wire_application_environment_calls_catalog_bootstrap_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    original = bootstrap_catalogs

    def _count(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(environment_wiring_module, "bootstrap_catalogs", _count)
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.once")
    wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)
    assert len(calls) == 1


def test_tool_load_failure_non_strict_isolated_in_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_tool_ep("broken", "not-a-valid-target")])
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.tool-fail")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    tools_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_TOOLS)
    assert tools_report is not None
    assert len(tools_report.failed) == 1
    assert tools_report.registered_count == 0
    assert tools_report.critical_bootstrap_acceptable is False


def test_skill_load_failure_non_strict_isolated_in_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_skill_ep("broken", "not-a-valid-target")])
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.skill-fail")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    skills_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SKILLS)
    assert skills_report is not None
    assert len(skills_report.failed) == 1
    assert skills_report.registered_count == 0
    assert skills_report.critical_bootstrap_acceptable is False


def test_strict_wire_application_environment_fails_on_tool_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_tool_ep("broken", "not-a-valid-target")])
    settings = LabApplicationSettings.from_env()
    env = _strict_env("stage10.strict-tool-fail")
    with pytest.raises(ToolCatalogBootstrapError):
        wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)


def test_strict_wire_application_environment_fails_on_skill_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_skill_ep("broken", "not-a-valid-target")])
    settings = LabApplicationSettings.from_env()
    env = _strict_env("stage10.strict-skill-fail")
    with pytest.raises(SkillCatalogBootstrapError):
        wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)


def test_rejected_tool_plugin_remains_non_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(
        monkeypatch,
        [
            _tool_ep("alpha", "_CollisionAlphaTool"),
            _tool_ep("beta", "_CollisionBetaTool"),
        ],
    )
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.tool-reject")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    tools_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_TOOLS)
    assert tools_report is not None
    assert tools_report.registered_count == 1
    assert len(tools_report.accepted) == 1
    assert len(tools_report.rejected) == 1
    assert tools_report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION
    assert _COLLISION_BUNDLE_ID in tool_catalog_snapshot()
    assert len([bid for bid in tool_catalog_snapshot() if bid == _COLLISION_BUNDLE_ID]) == 1


def test_rejected_invalid_tool_target_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_tool_ep("nope", "_NotATool")])
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.tool-invalid")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    tools_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_TOOLS)
    assert tools_report is not None
    assert tools_report.registered_count == 0
    assert len(tools_report.rejected) == 1
    assert tools_report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE


def test_rejected_invalid_skill_target_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(monkeypatch, [_skill_ep("nope", "_NotASkill")])
    settings = LabApplicationSettings.from_env()
    base_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.skill-invalid")
    env = base_env.model_copy(
        update={
            "capabilities": base_env.capabilities.model_copy(
                update={
                    "skills": SkillProfile(enabled=["stage10.skill.invalid.would-be"]),
                },
            ),
        },
    )
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    skills_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SKILLS)
    assert skills_report is not None
    assert skills_report.registered_count == 0
    assert len(skills_report.rejected) == 1
    assert skills_report.rejected[0].reason_code is PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    assert skills_report.rejected[0].spec.group == EP_SKILLS
    assert not any(
        "stage10.skill.invalid.would-be" in entry.skill_ids
        for entry in skill_catalog_snapshot().values()
    )
    assert not wiring.skill_wiring.registry.has("stage10.skill.invalid.would-be")


def test_rejected_skill_plugin_remains_non_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(
        monkeypatch,
        [
            _skill_ep("alpha", "_CollisionAlphaSkill"),
            _skill_ep("beta", "_CollisionBetaSkill"),
        ],
    )
    settings = LabApplicationSettings.from_env()
    base_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.skill-reject")
    env = base_env.model_copy(
        update={
            "capabilities": base_env.capabilities.model_copy(
                update={
                    "skills": SkillProfile(
                        enabled=[
                            "stage10.skill.collision.alpha",
                            "stage10.skill.collision.beta",
                        ],
                    ),
                },
            ),
        },
    )
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    skills_report = wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_SKILLS)
    assert skills_report is not None
    assert skills_report.registered_count == 1
    assert len(skills_report.accepted) == 1
    assert len(skills_report.rejected) == 1
    assert skills_report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION
    assert skills_report.rejected[0].spec.group == EP_SKILLS
    assert _SKILL_COLLISION_BUNDLE_ID in skill_catalog_snapshot()
    assert len([bid for bid in skill_catalog_snapshot() if bid == _SKILL_COLLISION_BUNDLE_ID]) == 1
    assert wiring.skill_wiring.registry.has("stage10.skill.collision.alpha")
    assert not wiring.skill_wiring.registry.has("stage10.skill.collision.beta")


def test_aggregate_domain_reports_include_tools_and_skills() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.aggregate")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    reports = wiring.platform_plugin_evidence.domain_reports
    assert PLATFORM_PLUGIN_DOMAIN_MEMORY in reports
    assert PLATFORM_PLUGIN_DOMAIN_CONTEXT in reports
    assert PLATFORM_PLUGIN_DOMAIN_SECURITY in reports
    assert PLATFORM_PLUGIN_DOMAIN_TOOLS in reports
    assert PLATFORM_PLUGIN_DOMAIN_SKILLS in reports


def test_bootstrap_catalogs_registered_count_with_mixed_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(INTERGRAX_DISCOVER_PLUGINS_ENV, "1")
    _install_eps(
        monkeypatch,
        [
            _tool_ep("good", "intergrax.tools.examples.custom_echo.plugin:CustomEchoToolPlugin"),
            _tool_ep("nope", "_NotATool"),
            _tool_ep("broken", "not-a-valid-target"),
        ],
    )
    result = bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    report = result.tool_plugin_load_report
    assert report.registered_count == 1
    assert len(report.accepted) == 1
    assert len(report.rejected) == 1
    assert len(report.failed) == 1


def test_platform_plugin_evidence_domain_reports_remain_immutable() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="stage10.immutable")
    wiring = wire_application_environment(build_lab_manifest(settings), env, conformance_check=False)

    with pytest.raises(TypeError):
        wiring.platform_plugin_evidence.domain_reports["tools"] = (
            wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_TOOLS)
        )
