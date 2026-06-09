# © Artur Czarnecki. All rights reserved.

"""ORCH-CONFIG.10 / CFG-11 — engine planner regression with mock LLM."""

from __future__ import annotations

import json

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _PlannerStubAgent(Agent):
    def __init__(self, *, agent_id: str, label: str) -> None:
        self._agent_id = agent_id
        self._label = label

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="engine planner gate stub",
            capabilities=[f"{self._agent_id}.run"],
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        return CapabilityMatchResult(matched=True, agent_id=self._agent_id, score=1.0)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        label = self._label

        class _Pipe(RuntimePipeline):
            async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
                answer = f"{label}: {state.request.message}"
                state.runtime_answer = RuntimeAnswer(run_id=state.run_id, answer=answer)
                return state.runtime_answer

        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="unused"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        config.pipeline = _Pipe()
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


@pytest.mark.asyncio
async def test_engine_planner_builds_multi_step_plan() -> None:
    registry = AgentRegistry()
    registry.register(_PlannerStubAgent(agent_id="planner_a", label="a"))
    registry.register(_PlannerStubAgent(agent_id="planner_b", label="b"))

    plan_json = json.dumps(
        {
            "steps": [
                {"agent_id": "planner_a", "description": "first", "depends_on": []},
                {"agent_id": "planner_b", "description": "second", "depends_on": ["llm_step_1"]},
            ]
        }
    )
    llm = FakeLLMAdapter(fixed_text=plan_json)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="engine.planner.gate").model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(planner_kind="engine"),
        }
    )
    loop = build_nexus_loop_from_environment(registry, env=env, llm_adapter=llm)
    result = await loop.handle_task(
        Task(
            tenant_id="org-gate",
            user_id="op-1",
            message="dynamic multi-agent task",
            context=TaskContext(capability="planner_a.run"),
        ),
    )
    assert result.state == TaskState.COMPLETED
    assert result.metadata.get("agent_ids") == ["planner_a", "planner_b"]
