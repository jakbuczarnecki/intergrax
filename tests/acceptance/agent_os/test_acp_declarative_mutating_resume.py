# © Artur Czarnecki. All rights reserved.

"""
ACP-PROD-1/2 acceptance — declarative mutating tool + checkpoint resume.

Ensures a committed idempotency key is replay-skipped after checkpoint resume
(no double mutating tool invoke).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore
from intergrax.agents.persistence.checkpoint_wiring import wire_acp_run_request
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.agents.persistence.tool_invoker_wiring import wire_acp_run_request_with_tool_invoker
from intergrax.contracts.acp_state import AcpSessionState
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel as ContractRiskLevel
from intergrax.contracts.agent_run import AgentExecutionOptions, AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, SideEffectMode, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]

TOOL_ID = "acp.acceptance.mutating_send"
IDEMPOTENCY_KEY = "acceptance:mutating:send:1"


class _In(BaseModel):
    payload: str = ""


class _Out(BaseModel):
    sent: bool = True


_MUTATING_TOOL = ToolContract(
    tool_id=TOOL_ID,
    name=TOOL_ID,
    description="acceptance mutating send",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=True,
    risk_level=ToolRiskLevel.HIGH,
)


class _DeclarativeMutatingResumeProbe(IntergraxAgent):
    contract_id = "declarative_mutating_resume_probe"
    capabilities = ("harness.acp.declarative_mutating",)
    agent_name = "Declarative Mutating Resume Probe"
    risk_level = ContractRiskLevel.HIGH
    max_steps = 4
    session_state_type = AcpSessionState

    def get_contract(self) -> AgentContract:
        contract = super().get_contract()
        return contract.model_copy(update={"extra_tools": [_MUTATING_TOOL]})

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        state = self.load_session_state(step_ctx)
        phase = state.phase or "send"
        action = {
            "tool_id": TOOL_ID,
            "idempotency_key": IDEMPOTENCY_KEY,
            "args": {"payload": "acceptance"},
        }
        if phase == "send":
            return StepOutcome.continue_with({"phase": "retry_pending"}).model_copy(
                update={"requested_actions": [action]},
            )
        if phase == "retry_pending":
            return StepOutcome.continue_with({"phase": "done"}).model_copy(
                update={"requested_actions": [action]},
            )
        return StepOutcome.complete(
            {"status": "ok"},
            terminal_reason=TerminalReason.GOAL_MET,
        )


@pytest.mark.asyncio
async def test_acceptance_05d_acp_declarative_mutating_resume() -> None:
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-acceptance")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    agent = _DeclarativeMutatingResumeProbe()
    store = InMemoryAgentCheckpointStore()
    run_id = "acceptance-acp-decl-mutating-1"

    base = AgentRunRequest(
        input="declarative-mutating-resume",
        identity=RequestIdentity(tenant_id="t-agent-os", user_id="u-acp"),
        metadata={"run_id": run_id, "user_id": "u-acp"},
        execution_options=AgentExecutionOptions(
            side_effect_mode=SideEffectMode.DECLARATIVE,
            checkpoint_every_step=True,
        ),
    )
    base = wire_acp_run_request_with_tool_invoker(base, invoker)

    await agent.run(
        wire_acp_run_request(
            base.model_copy(
                update={
                    "execution_options": AgentExecutionOptions(
                        max_steps=1,
                        side_effect_mode=SideEffectMode.DECLARATIVE,
                        checkpoint_every_step=True,
                    ),
                },
            ),
            store,
        ),
    )
    checkpoint = store.get_latest(run_id, "t-agent-os")
    assert checkpoint is not None
    assert invoke_count == 1
    assert any(
        record.status.value == "committed" and record.idempotency_key == IDEMPOTENCY_KEY
        for record in checkpoint.side_effect_ledger
    )

    result = await agent.run(
        wire_acp_run_request(
            base.model_copy(
                update={
                    "execution_options": AgentExecutionOptions(
                        max_steps=10,
                        side_effect_mode=SideEffectMode.DECLARATIVE,
                        checkpoint_every_step=True,
                    ),
                },
            ),
            store,
            resume=True,
        ),
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert invoke_count == 1
