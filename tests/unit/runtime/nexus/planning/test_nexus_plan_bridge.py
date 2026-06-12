# © Artur Czarnecki. All rights reserved.

import json

import pytest

from intergrax.contracts.reasoning_failure import ReasoningFailureKind
from intergrax.runtime.nexus.planning.nexus_plan_bridge import (
    build_nexus_plan_unified,
    build_planner_build_debug,
)
from intergrax.runtime.nexus.planning.task_planner import TaskPlanner
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_planner_build_debug_includes_source() -> None:
    payload = build_planner_build_debug(planner_source="engine", used_fallback=True)
    assert payload["planner_source"] == "engine"
    assert payload["used_fallback"] == "true"


@pytest.mark.asyncio
async def test_build_nexus_plan_unified_parses_llm_json() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    agent_id = registry.list_agent_ids()[0]
    llm = FakeLLMAdapter(
        fixed_text=json.dumps(
            {"steps": [{"agent_id": agent_id, "description": "step", "depends_on": []}]}
        )
    )
    task = Task(tenant_id="t", user_id="u", message="hi", context=TaskContext())
    plan, debug = build_nexus_plan_unified(
        task,
        registry,
        llm,
        fallback=TaskPlanner(),
        prompt_text="plan",
        planner_source="engine",
    )
    assert len(plan.steps) == 1
    assert debug.used_fallback is False
    assert plan.plan_metadata.get("planner_source") == "engine"


def test_build_nexus_plan_unified_fallback_on_bad_json() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    llm = FakeLLMAdapter(fixed_text="not json")
    task = Task(tenant_id="t", user_id="u", message="hi", context=TaskContext())
    _plan, debug = build_nexus_plan_unified(
        task,
        registry,
        llm,
        fallback=TaskPlanner(),
        prompt_text="plan",
    )
    assert debug.failure_kind is ReasoningFailureKind.PLANNER_PARSE_FAILED


def test_build_nexus_plan_unified_parse_retries_recover() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    agent_id = registry.list_agent_ids()[0]
    good = json.dumps(
        {"steps": [{"agent_id": agent_id, "description": "step", "depends_on": []}]}
    )

    class RetryLLM(FakeLLMAdapter):
        def __init__(self) -> None:
            super().__init__(fixed_text="bad")
            self._calls = 0

        def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
            self._calls += 1
            if self._calls == 1:
                return super().generate_messages(
                    messages, temperature=temperature, max_tokens=max_tokens, run_id=run_id
                )
            self._fixed_text = good
            return super().generate_messages(
                messages, temperature=temperature, max_tokens=max_tokens, run_id=run_id
            )

    llm = RetryLLM()
    task = Task(tenant_id="t", user_id="u", message="hi", context=TaskContext())
    plan, debug = build_nexus_plan_unified(
        task,
        registry,
        llm,
        fallback=TaskPlanner(),
        prompt_text="plan",
        planner_parse_retries=1,
    )
    assert len(plan.steps) == 1
    assert debug.used_fallback is False
    assert llm._calls == 2
