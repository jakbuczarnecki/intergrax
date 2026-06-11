# © Artur Czarnecki. All rights reserved.

"""ORCH-CONFIG.10 / CFG-11 — engine planner regression with mock LLM."""

from __future__ import annotations

import json

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from testing_support.builder import FakeLLMAdapter
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_engine_planner_builds_multi_step_plan() -> None:
    registry = AgentRegistry()
    registry.register(
        UaepPipelineStubAgent(
            agent_id="planner_a",
            capability="planner_a.run",
            prefix="a",
            always_match=True,
            description="engine planner gate stub",
        )
    )
    registry.register(
        UaepPipelineStubAgent(
            agent_id="planner_b",
            capability="planner_b.run",
            prefix="b",
            always_match=True,
            description="engine planner gate stub",
        )
    )

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
