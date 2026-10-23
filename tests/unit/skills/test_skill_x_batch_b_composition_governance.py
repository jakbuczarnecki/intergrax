# © Artur Czarnecki. All rights reserved.

"""SKILL-X Batch B — composition, tool scope, ACP metadata, registry integration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.tool_selection import (
    SkillPackSelectionStrategy,
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.skills.core.contracts import SkillManifest
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.errors import ApplicationManifestConformanceError
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.application_composition_context import (
    composition_for_factory_context,
)
from intergrax.skills.integration.contract_resolution import (
    eligible_agent_skill_manifests,
    resolve_contract_tools,
    resolve_gateway_allowed_tools,
)
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolutionError, SkillResolver
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.builder import tools_agent_make_contract

pytestmark = pytest.mark.unit


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


def _tool(tool_id: str) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=tool_id,
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
    )


_SKILL_A = SkillManifest(
    skill_id="a.pack",
    description="a",
    tool_ids=("tool.a",),
)
_SKILL_B = SkillManifest(
    skill_id="b.pack",
    description="b",
    tool_ids=("tool.b",),
)


def _contract_with_skills(*skills: SkillManifest) -> AgentContract:
    return AgentContract(
        id="batch-b-agent",
        name="Batch B",
        description="batch b tests",
        capabilities=["batch.b"],
        skills=list(skills),
    )


def _registry_with_skills() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(_SKILL_A)
    registry.register(_SKILL_B)
    return registry


def test_profile_free_context_passes_agent_skills_without_profile() -> None:
    eligible = eligible_agent_skill_manifests((_SKILL_A, _SKILL_B), None)
    assert eligible == (_SKILL_A, _SKILL_B)


def test_disabled_skill_fails_closed_when_profile_provided() -> None:
    skills = _registry_with_skills()
    contract = _contract_with_skills(_SKILL_A, _SKILL_B)
    resolver = SkillResolver(skills)
    profile = SkillProfile(enabled=["a.pack"])
    with pytest.raises(SkillResolutionError, match="not enabled"):
        resolve_contract_tools(
            contract,
            skill_resolver=resolver,
            skill_profile=profile,
        )


def test_enabled_skill_resolves_and_disabled_skill_omitted_from_pack() -> None:
    skills = _registry_with_skills()
    contract = _contract_with_skills(_SKILL_A)
    resolver = SkillResolver(skills)
    profile = SkillProfile(enabled=["a.pack"])
    resolved, pack = resolve_contract_tools(
        contract,
        skill_resolver=resolver,
        skill_profile=profile,
    )
    assert resolved.allowed_tools == ["tool.a"]
    assert pack.skill_ids == ("a.pack",)


def test_tool_only_agent_registry_without_skill_resolution() -> None:
    registry = AgentRegistry()
    contract = AgentContract(
        id="tool-only",
        name="Tool Only",
        description="tool only",
        capabilities=["tool.only"],
        allowed_tools=["direct.tool"],
    )

    class _Stub:
        def get_contract(self) -> AgentContract:
            return contract

    registry.register(_Stub(), contract=contract)
    assert registry.get_contract("tool-only").allowed_tools == ["direct.tool"]
    assert registry.get_resolved_skill_pack("tool-only") is None


def test_skill_tool_candidates_clamped_to_host_tool_profile() -> None:
    skills = _registry_with_skills()
    contract = _contract_with_skills(_SKILL_A)
    resolver = SkillResolver(skills)
    profile = SkillProfile(enabled=["a.pack"])
    tool_profile = ToolProfile(enabled=["tool.other"])
    with pytest.raises(SkillResolutionError, match="exceed host ToolProfile"):
        resolve_contract_tools(
            contract,
            skill_resolver=resolver,
            skill_profile=profile,
            tool_profile=tool_profile,
        )


def test_agent_registry_passes_skill_and_tool_profile() -> None:
    skills = _registry_with_skills()
    contract = _contract_with_skills(_SKILL_A)
    tool_profile = ToolProfile(enabled=["tool.a"])
    registry = AgentRegistry()

    class _Stub:
        def get_contract(self) -> AgentContract:
            return contract

    registry.register(
        _Stub(),
        contract=contract,
        skill_registry=skills,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        tool_profile=tool_profile,
    )
    assert registry.get_contract("batch-b-agent").allowed_tools == ["tool.a"]


def test_gateway_absent_metadata_preserves_contract_ceiling() -> None:
    assert resolve_gateway_allowed_tools(["a.tool", "b.tool"], None) == [
        "a.tool",
        "b.tool",
    ]


def test_gateway_explicit_empty_metadata_denies_all() -> None:
    assert resolve_gateway_allowed_tools(["a.tool", "b.tool"], []) == []


def test_gateway_malformed_metadata_fails_closed() -> None:
    with pytest.raises(SkillResolutionError, match="must be a list"):
        resolve_gateway_allowed_tools(["a.tool"], "not-a-list")


def test_gateway_metadata_cannot_widen_contract_tools() -> None:
    assert resolve_gateway_allowed_tools(["allowed.tool"], ["evil.tool"]) == []


def test_gateway_empty_contract_ceiling_blocks_metadata_widening() -> None:
    assert resolve_gateway_allowed_tools([], ["evil.tool"]) == []


def test_gateway_metadata_may_narrow_contract_tools() -> None:
    resolved = resolve_gateway_allowed_tools(
        ["a.tool", "b.tool"],
        ["b.tool"],
    )
    assert resolved == ["b.tool"]


class _EchoHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        return _Out()


def _tool_registry_with(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        registry.register(tools_agent_make_contract(tool_id, _In, _Out), _EchoHandler())
    return registry


def test_register_binding_requires_skill_profile_when_composition_present() -> None:
    from intergrax.agents.echo.echo_agent import EchoAgent
    from intergrax.applications._shared.roster_agent_contract_authority import (
        ManifestAgentContractAuthority,
    )
    from intergrax.applications._shared.wiring import _register_binding

    manifest = ApplicationManifest.lab(
        app_id="batch_b_skill_gate",
        name="Batch B Skill Gate",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo")],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    composition = composition_for_factory_context(ctx)
    authority = ManifestAgentContractAuthority(
        contracts_by_id={"echo": _contract_with_skills(_SKILL_A)},
    )
    registry = AgentRegistry()
    with pytest.raises(ApplicationManifestConformanceError, match="skill_profile required"):
        _register_binding(
            registry,
            manifest.agents[0],
            ctx,
            builders=None,
            skill_registry=None,
            composition=composition,
            roster_contract_authority=authority,
        )


def test_environment_wiring_carries_host_skill_profile_in_composition() -> None:
    from intergrax.applications._shared.environment_wiring import wire_application_environment
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )
    from lab_application.host.settings import LabApplicationSettings
    from lab_application.manifest import build_lab_manifest

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="batch_b_env_skill")
    manifest = build_lab_manifest(settings)
    wiring = wire_application_environment(
        manifest,
        env,
        settings=settings,
        conformance_check=False,
    )
    assert wiring.composition.skill_profile is env.skill_profile


def test_skill_pack_strategy_respects_plan_allowed_tool_ids() -> None:
    tools = _tool_registry_with("tool.a", "tool.b")
    skills = _registry_with_skills()
    profile = SkillProfile(enabled=["a.pack", "b.pack"])
    ctx = ToolSelectionContext(
        registry=tools,
        query="",
        skill_profile=profile,
        skill_registry=skills,
        plan_allowed_tool_ids=("tool.a",),
    )
    strategy = SkillPackSelectionStrategy()
    ids = resolve_planner_allowed_tool_ids(ToolSelectionMode.SKILL_PACK, ctx)
    assert ids == ("tool.a",)
