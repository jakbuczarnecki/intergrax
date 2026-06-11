# © Artur Czarnecki. All rights reserved.

"""
ACP-CLOSE-PROD-4 — Nexus + harness host catalog declarative invoker E2E.

Exercises ``build_harness_host_runtime`` → ``NexusLoop`` → ``CatalogDeclarativeToolInvoker``
with checkpoint resume (no ``CallableDeclarativeToolInvoker`` mock).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.persistence.catalog_declarative_invoker import (
    build_catalog_declarative_invoker_from_registry,
)
from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.acp_state import AcpSessionState
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel as ContractRiskLevel
from intergrax.contracts.agent_run_enums import SideEffectMode, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]

TOOL_ID = "acp.acceptance.mutating_send"
IDEMPOTENCY_KEY = "acceptance:nexus:catalog:send:1"


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


class _MutatingSendHandler(ToolHandler[_In, _Out]):
    invoke_count = 0

    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _MutatingSendHandler.invoke_count += 1
        return _Out(sent=True)


class _NexusCatalogDeclarativeProbe(IntergraxAgent):
    contract_id = "nexus_catalog_declarative_probe"
    capabilities = ("harness.acp.declarative_mutating",)
    agent_name = "Nexus Catalog Declarative Probe"
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
            "args": {"payload": "nexus-acceptance"},
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


def _catalog_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_MUTATING_TOOL, _MutatingSendHandler())
    return registry


def _nexus_task(*, run_id: str, max_steps: int) -> Task:
    return Task(
        task_id=run_id,
        tenant_id="t-agent-os",
        user_id="u-acp",
        message="nexus-catalog-declarative-resume",
        context=TaskContext(capability="harness.acp.declarative_mutating"),
        metadata={
            AcpMetadataKey.SESSION_ENABLED: True,
            "user_id": "u-acp",
            "run_id": run_id,
            "acp.execution_options.v1": {
                "max_steps": max_steps,
                "side_effect_mode": SideEffectMode.DECLARATIVE.value,
                "checkpoint_every_step": True,
            },
        },
    )


@pytest.mark.asyncio
async def test_acceptance_05e_nexus_harness_catalog_declarative_mutating_resume() -> None:
    _MutatingSendHandler.invoke_count = 0
    tool_registry = _catalog_tool_registry()
    catalog_invoker = build_catalog_declarative_invoker_from_registry(tool_registry)

    registry = AgentRegistry()
    registry.register(_NexusCatalogDeclarativeProbe())

    agent_store = InMemoryAgentCheckpointStore()
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment or build_lab_environment_profile(settings)
    run_id = "acceptance-nexus-catalog-decl-1"

    with patch(
        "intergrax.applications._shared.harness_host_runtime.build_declarative_invoker_from_tool_wiring",
        return_value=catalog_invoker,
    ):
        runtime = build_harness_host_runtime(
            manifest.model_copy(update={"environment": env}),
            env,
            settings=settings,
            registry=registry,
            agent_checkpoint_store=agent_store,
            use_in_memory_trace=True,
        )

    nexus = runtime.nexus_loop
    assert runtime.nexus_loop._declarative_tool_invoker is catalog_invoker  # noqa: SLF001

    await nexus.handle_task(_nexus_task(run_id=run_id, max_steps=1))
    checkpoint = agent_store.get_latest(run_id, "t-agent-os")
    assert checkpoint is not None
    assert _MutatingSendHandler.invoke_count == 1
    assert any(
        record.status.value == "committed" and record.idempotency_key == IDEMPOTENCY_KEY
        for record in checkpoint.side_effect_ledger
    )

    result = await nexus.handle_task(_nexus_task(run_id=run_id, max_steps=10))
    assert result.state == TaskState.COMPLETED
    assert _MutatingSendHandler.invoke_count == 1
