# © Artur Czarnecki. All rights reserved.

import pytest
from pydantic import BaseModel

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from testing_support.catalog_declarative_invoker import (
    build_catalog_declarative_invoker_from_registry,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from testing_support.builder import tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int


class _EchoHandler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value + 1)


TOOL_ID = "acp.echo_tool"


def _bind_catalog_execution(run_id: str) -> tuple[object, object]:
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(RunBudget()),
    )
    return identity_token, budget_token


def _registry_with_tool() -> ToolRegistry:
    registry = ToolRegistry()
    contract = tools_agent_make_contract(TOOL_ID, _In, _Out)
    registry.register(contract, _EchoHandler())
    return registry


@pytest.mark.asyncio
async def test_catalog_declarative_invoker_routes_through_catalog() -> None:
    registry = _registry_with_tool()
    invoker = build_catalog_declarative_invoker_from_registry(registry)
    run_id = mint_run_id()
    identity_token, budget_token = _bind_catalog_execution(run_id)
    try:
        result = await invoker.invoke(
            tenant_id="tenant-1",
            run_id=str(run_id),
            task_id=str(mint_task_id()),
            agent_id="agent-a",
            tool_id=TOOL_ID,
            args={"value": 4},
            idempotency_key="key-1",
        )
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)
    assert result.status == "success"
    assert result.output == {"result": 5}


def test_catalog_declarative_invoker_builds_real_runtime_context() -> None:
    registry = _registry_with_tool()
    invoker = build_catalog_declarative_invoker_from_registry(registry)
    run_id = mint_run_id()
    task_id = mint_task_id()
    state = invoker._runtime_state(  # noqa: SLF001 — wiring verification
        tenant_id="tenant-1",
        run_id=str(run_id),
        task_id=str(task_id),
        agent_id="agent-a",
        user_id="",
    )
    assert isinstance(state.context.session_manager, SessionManager)
    assert isinstance(state.context.config.llm_adapter, LLMAdapter)


@pytest.mark.asyncio
async def test_catalog_declarative_invoker_bind_run_sets_session_user_id() -> None:
    registry = _registry_with_tool()
    invoker = CatalogDeclarativeToolInvoker(
        tool_invoker=build_catalog_declarative_invoker_from_registry(registry).tool_invoker,
    )
    invoker.bind_run(user_id="user-session-1")
    assert invoker.binding.user_id == "user-session-1"
