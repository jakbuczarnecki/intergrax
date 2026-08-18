# © Artur Czarnecki. All rights reserved.

"""Declarative policy REQUIRE_HITL → canonical Nexus HITL E2E (ADR-PLATFORM-PLUGIN-001)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from intergrax.utils import attribute_access
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TOOL_ID = "hitl.governed.tool"
_RULE_ID = "hitl.governed.tool"


class _Input(BaseModel):
    value: int = 1


class _Output(BaseModel):
    value: int


class _CountingExecutor(ToolExecutor):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return _Output(value=request.input.value)


def _policy_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.hitl.nexus")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": _RULE_ID,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _build_runtime_context(request: RuntimeRequest, executor: _CountingExecutor) -> RuntimeContext:
    contract = ToolContract(
        tool_id=_TOOL_ID,
        name=_TOOL_ID,
        description="hitl e2e",
        input_schema=_Input,
        output_schema=_Output,
        side_effects=True,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )
    registry = FakeRegistry(contract)
    invoker = RuntimeToolInvoker(registry=registry, executor=executor, scope_policy=None)
    bundle = _policy_bundle()
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(fixed_text="ok"),
        enable_rag=False,
        production_mode=False,
        tenant_id=request.tenant_id,
        tool_invoker=invoker,
    )
    config.policy_bundle = bundle
    context = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    context.config.tool_invoker = invoker
    context.config.tool_registry = registry
    context.config.policy_bundle = bundle
    return context


class _PolicyHitlToolAgent(Agent):
    def __init__(self, executor: _CountingExecutor) -> None:
        self._executor = executor

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="policy_hitl_tool",
            name="Policy HITL Tool Agent",
            description="invokes governed tool",
            capabilities=["policy.hitl.tool"],
            allowed_tools=[_TOOL_ID],
            max_steps=1,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability in (None, "policy.hitl.tool"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="policy_hitl_tool",
                matched_capabilities=["policy.hitl.tool"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="unsupported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _build_runtime_context(request, self._executor)

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="invoke_tool", step_name="invoke", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        await ctx.invoke_tool(
            ToolRequest(
                tool_name=_TOOL_ID,
                agent_id=ctx.agent_id,
                step_id=step.step_id,
                input={"value": 1},
                idempotency_key="hitl-e2e-idem",
            )
        )
        return StepOutput(step_id=step.step_id, summary="tool invoked")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")


@pytest.mark.asyncio
async def test_declarative_policy_hitl_nexus_pause_approve_resume(tmp_path) -> None:
    executor = _CountingExecutor()
    registry = AgentRegistry()
    registry.register(_PolicyHitlToolAgent(executor))
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "hitl_ckpt.db")
    loop = NexusLoop(registry, checkpoint_store=store)

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="run governed tool",
        context=TaskContext(capability="policy.hitl.tool"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )

    paused = await loop.handle_task(task)
    assert executor.calls == 0
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert paused.metadata.get("governance_human_request") is not None
    assert paused.execution_result is not None
    assert paused.execution_result.status == AgentExecutionStatus.NEEDS_INPUT
    assert paused.execution_result.declarative_hitl_pending is not None
    assert paused.execution_result.declarative_hitl_pending.tool_id == _TOOL_ID
    token = paused.summary.resume_token
    assert token

    approved = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="run governed tool",
            context=TaskContext(capability="policy.hitl.tool"),
            task_id=paused.task_id,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(
                    enabled=True,
                    resume_token=token,
                ),
            ),
            metadata={"human_response": "approve", "resume_token": token},
        )
    )

    assert executor.calls == 1
    assert approved.state == TaskState.COMPLETED
    assert approved.execution_result is not None
    assert approved.execution_result.declarative_hitl_pending is None
