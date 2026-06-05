# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.runtime.registry.agent_assembly_resolver import (
    AgentAssemblyError,
    assert_agent_assembly_valid,
    validate_agent_assembly,
    validate_contract_metadata,
    validate_lifecycle_metadata,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.skills.core.contracts import SkillManifest
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


_TOOL = ToolContract(
    tool_id="demo.tool",
    name="demo.tool",
    description="demo",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=False,
    risk_level=ToolRiskLevel.LOW,
)

_SKILL = SkillManifest(
    skill_id="demo.skill",
    description="demo skill",
    tool_ids=("demo.tool",),
)


def _valid_contract(**updates: object) -> AgentContract:
    base = AgentContract(
        id="demo_agent",
        name="Demo Agent",
        description="Demo agent for assembly validation.",
        capabilities=["demo.cap"],
        skills=[_SKILL],
        extra_tools=[_TOOL],
    )
    return base.model_copy(update=updates)


@pytest.mark.unit
@pytest.mark.gate
def test_validate_contract_metadata_requires_capability_ids() -> None:
    result = validate_contract_metadata(_valid_contract(capabilities=[]))
    assert not result.valid
    assert any("capabilities" in error for error in result.errors)


@pytest.mark.unit
@pytest.mark.gate
def test_validate_contract_metadata_rejects_author_time_allowed_tools() -> None:
    result = validate_contract_metadata(
        _valid_contract(allowed_tools=["demo.tool"]),
    )
    assert not result.valid
    assert any("allowed_tools" in error for error in result.errors)


@pytest.mark.unit
@pytest.mark.gate
def test_validate_lifecycle_metadata_requires_owner_for_production_eligible() -> None:
    result = validate_lifecycle_metadata(
        _valid_contract(
            production_eligible=True,
            owner_team="",
            owner_contact="",
            runbook_ref="",
        ),
    )
    assert not result.valid
    assert len(result.errors) == 3


@pytest.mark.unit
@pytest.mark.gate
def test_validate_lifecycle_metadata_accepts_production_eligible_with_owner() -> None:
    result = validate_lifecycle_metadata(
        _valid_contract(
            production_eligible=True,
            owner_team="platform",
            owner_contact="owner@intergrax",
            runbook_ref="docs/INTERGRAX_IMPLEMENTATION_PLAN.md",
        ),
    )
    assert result.valid


@pytest.mark.unit
@pytest.mark.gate
def test_assert_agent_assembly_valid_raises_on_invalid_contract() -> None:
    with pytest.raises(AgentAssemblyError, match="capabilities"):
        assert_agent_assembly_valid(_valid_contract(capabilities=[]))


@pytest.mark.unit
@pytest.mark.gate
def test_agent_registry_runs_assembly_validation_at_register() -> None:
    from intergrax.agents.agent_contract import Agent
    from intergrax.contracts.capability import CapabilityMatchResult
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
    from intergrax.runtime.task.task import TaskContext
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    class _InvalidAgent(Agent):
        def get_contract(self) -> AgentContract:
            return AgentContract(
                id="bad",
                name="Bad",
                description="missing capabilities",
                capabilities=[],
            )

        def build_context(self, request: RuntimeRequest) -> RuntimeContext:
            config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
            return RuntimeContext.build(
                config=config,
                session_manager=build_in_memory_session_manager(),
            )

        def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
            return CapabilityMatchResult(matched=False)

    registry = AgentRegistry()
    with pytest.raises(AgentAssemblyError):
        registry.register(_InvalidAgent())


@pytest.mark.unit
@pytest.mark.gate
def test_validate_agent_assembly_accepts_retired_lifecycle_for_registry_introspection() -> None:
    result = validate_agent_assembly(
        _valid_contract(lifecycle_state=AgentLifecycleState.RETIRED),
    )
    assert result.valid
