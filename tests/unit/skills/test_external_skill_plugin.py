# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from intergrax.skills.examples.custom_pack import CustomPackSkillPlugin
from intergrax.skills.examples.custom_pack.plugin import CUSTOM_PACK_SKILL_ID
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.skills.registry.factory import build_registry_from_profile as build_skill_registry_from_profile
from intergrax.skills.registry.plugin_register import register_skill_plugin
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.examples.custom_echo import CustomEchoToolPlugin
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog
from intergrax.tools.registry.factory import build_registry_from_profile as build_tool_registry_from_profile
from intergrax.tools.registry.plugin_register import register_tool_plugin
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


class _PackAgent(Agent):
    def get_contract(self) -> AgentContract:
        manifests = CustomPackSkillPlugin.skill_manifests()
        return AgentContract(
            id="pack_stub",
            name="Pack Stub",
            description="stub",
            capabilities=["stub.cap"],
            skills=list(manifests),
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        return CapabilityMatchResult(matched=True, agent_id="pack_stub", score=1.0)


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_skill_catalog()
    clear_tool_catalog()
    reset_default_skills_for_tests()
    reset_default_tools_bootstrap()
    yield
    clear_skill_catalog()
    clear_tool_catalog()
    reset_default_skills_for_tests()
    reset_default_tools_bootstrap()


def test_external_skill_plugin_merges_allowed_tools() -> None:
    register_skill_plugin(CustomPackSkillPlugin)
    register_tool_plugin(CustomEchoToolPlugin)
    skill_registry = build_skill_registry_from_profile(SkillProfile(enabled_bundles=["custom_pack"]))
    tool_registry = build_tool_registry_from_profile(
        ToolProfile(enabled_bundles=["custom_echo"]),
        ctx=ToolWiringContext(),
    )

    registry = AgentRegistry()
    registry.register(
        _PackAgent(),
        skill_registry=skill_registry,
        tool_registry=tool_registry,
    )
    contract = registry.get_contract("pack_stub")
    assert "custom_echo.ping" in contract.allowed_tools
