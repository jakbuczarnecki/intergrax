# © Artur Czarnecki. All rights reserved.

"""
ACP-CLOSE-PROD-6 acceptance — cross-run idempotency store dedupe.

Ensures a committed idempotency key is replay-skipped on a new run_id when
``ReliabilityProfile.idempotency_store`` is wired (not only checkpoint resume).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.persistence.idempotency_store_wiring import (
    wire_acp_run_request_with_idempotency_store,
)
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
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]

TOOL_ID = "acp.acceptance.cross_run_send"
IDEMPOTENCY_KEY = "acceptance:cross-run:send:1"
TENANT = "t-agent-os-cross"


class _In(BaseModel):
    payload: str = ""


class _Out(BaseModel):
    sent: bool = True


_MUTATING_TOOL = ToolContract(
    tool_id=TOOL_ID,
    name=TOOL_ID,
    description="acceptance cross-run mutating send",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=True,
    risk_level=ToolRiskLevel.HIGH,
)


class _CrossRunDedupeProbe(IntergraxAgent):
    contract_id = "cross_run_dedupe_probe"
    capabilities = ("harness.acp.cross_run_dedupe",)
    agent_name = "Cross Run Dedupe Probe"
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
            "args": {"payload": "cross-run"},
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
async def test_acceptance_05e_acp_declarative_cross_run_idempotency_dedupe() -> None:
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-cross-run")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    agent = _CrossRunDedupeProbe()
    idempotency_store = InMemoryIdempotencyStore()

    def _request(run_id: str, max_steps: int) -> AgentRunRequest:
        base = AgentRunRequest(
            input="declarative-cross-run-dedupe",
            identity=RequestIdentity(tenant_id=TENANT, user_id="u-acp"),
            metadata={"run_id": run_id, "user_id": "u-acp"},
            execution_options=AgentExecutionOptions(
                max_steps=max_steps,
                side_effect_mode=SideEffectMode.DECLARATIVE,
                checkpoint_every_step=False,
            ),
        )
        wired = wire_acp_run_request_with_tool_invoker(base, invoker)
        return wire_acp_run_request_with_idempotency_store(wired, idempotency_store)

    await agent.run(_request("acceptance-cross-run-1", max_steps=1))
    assert invoke_count == 1

    result = await agent.run(_request("acceptance-cross-run-2", max_steps=10))
    assert result.status == AgentRunStatus.SUCCEEDED
    assert invoke_count == 1
