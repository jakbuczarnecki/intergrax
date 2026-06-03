# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.task.task import TaskContext
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


_EXTRA_TOOL = ToolContract(
    tool_id="extra.tool",
    name="extra.tool",
    description="extra",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=False,
)

_DEMO_SKILL = SkillManifest(
    skill_id="demo.pack",
    description="demo",
    tool_ids=("rag.retrieve",),
)


class _SkillAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="skill_stub",
            name="Skill Stub",
            description="stub",
            capabilities=["stub.cap"],
            skills=[_DEMO_SKILL],
            extra_tools=[_EXTRA_TOOL],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        return CapabilityMatchResult(matched=True, agent_id="skill_stub", score=1.0)


@pytest.mark.unit
def test_agent_registry_resolves_skills_into_allowed_tools() -> None:
    skills = SkillRegistry()
    skills.register(_DEMO_SKILL)
    registry = AgentRegistry()
    registry.register(_SkillAgent(), skill_registry=skills)
    contract = registry.get_contract("skill_stub")
    assert "rag.retrieve" in contract.allowed_tools
    assert "extra.tool" in contract.allowed_tools


@pytest.mark.unit
def test_agent_registry_unknown_skill_raises() -> None:
    from intergrax.skills.resolver import SkillResolutionError

    registry = AgentRegistry()
    with pytest.raises(SkillResolutionError, match="Unknown skill_id"):
        registry.register(_SkillAgent())
