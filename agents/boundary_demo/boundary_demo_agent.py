# © Artur Czarnecki. All rights reserved.

"""UAEP agent that stores a demo record via ``records.put`` (partner PoC)."""

from __future__ import annotations

from typing import Any

from boundary_demo.capabilities import CAPABILITIES, CAPABILITY
from intergrax.agents.agent_contract import Agent
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_config,
    default_reference_harness,
)
from intergrax.agents.tool_enablement import ToolEnablementProfile, ToolWiringContextLike
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.settings import ExecutionBoundaryExportRuntimeSettings
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.task.task import TaskContext
from intergrax.tools.providers.records.contracts import RecordsPutInput
from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter
from intergrax.tools.providers.records.service import RECORDS_PUT_TOOL_ID


class BoundaryDemoAgent(Agent):
    """Single-step UAEP agent for Execution Boundary Export partner sandbox."""

    AGENT_ID = "boundary_demo_agent"

    def __init__(
        self,
        harness: LabHarnessContext | None = None,
        *,
        tool_profile: ToolEnablementProfile | None = None,
        tool_wiring_context: ToolWiringContextLike | None = None,
        execution_boundary_export: ExecutionBoundaryExportRuntimeSettings | None = None,
        boundary_event_buffer: BoundaryEventBuffer | None = None,
    ) -> None:
        self._harness = harness or default_reference_harness()
        self._tool_profile = tool_profile
        self._tool_wiring_context = tool_wiring_context
        self._execution_boundary_export = execution_boundary_export
        self._boundary_event_buffer = boundary_event_buffer

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.AGENT_ID,
            name="Boundary Demo Agent",
            description="Partner PoC agent — writes a demo record via records.put.",
            version="0.1.0",
            capabilities=list(CAPABILITIES),
            allowed_tools=[RECORDS_PUT_TOOL_ID],
            skills=[],
            extra_tools=[],
            risk_level=AgentRiskLevel.MEDIUM,
            lifecycle_state=AgentLifecycleState.STAGING,
            owner_team="platform",
            max_steps=1,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, CAPABILITY):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.AGENT_ID,
                matched_capabilities=[CAPABILITY],
                score=1.0,
                rationale="attestation demo capability",
            )
        return CapabilityMatchResult(matched=False, rationale="unsupported capability")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = build_lab_agent_runtime_config(
            request=request,
            llm_adapter=PrefixStubLLMAdapter(prefix="boundary-demo"),
            harness=self._harness,
            enable_rag=False,
            enable_websearch=False,
        )
        config.tool_profile = self._tool_profile
        config.tool_wiring_context = self._tool_wiring_context
        if self._execution_boundary_export is not None:
            config.execution_boundary_export = self._execution_boundary_export
        if self._boundary_event_buffer is not None:
            config.boundary_event_buffer = self._boundary_event_buffer
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [
            AgentStep(
                step_id="store_demo_record",
                step_name="store_demo_record",
                step_index=0,
                allowed_tools=[RECORDS_PUT_TOOL_ID],
            )
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        metadata = dict(ctx.request.metadata) if ctx.request is not None else {}
        partition_key = str(metadata.get("partition_key") or "attestation_demo")
        row_key = str(metadata.get("row_key") or f"poc-{ctx.run_id[:12]}")
        record_data = metadata.get("record_data")
        if not isinstance(record_data, dict):
            message = (ctx.request.message if ctx.request else "") or "PoC report"
            record_data = {"title": message, "version": 1}

        payload = RecordsPutInput(
            partition_key=partition_key,
            row_key=row_key,
            data=record_data,
        )
        response = await ctx.invoke_tool(
            ToolRequest(
                tool_name=RECORDS_PUT_TOOL_ID,
                agent_id=ctx.agent_id,
                step_id=step.step_id,
                input=payload.model_dump(),
            )
        )
        if response.status != ToolResponseStatus.SUCCESS:
            raise RuntimeError(response.error or "records.put failed")

        output_data: dict[str, Any] = {
            "stored": True,
            "partition_key": partition_key,
            "row_key": row_key,
            "tool_output": response.output,
        }
        return StepOutput(
            step_id=step.step_id,
            summary=f"stored record {partition_key}/{row_key}",
            data=output_data,
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="demo record stored")
