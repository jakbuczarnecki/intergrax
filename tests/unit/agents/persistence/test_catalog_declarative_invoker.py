# © Artur Czarnecki. All rights reserved.

import pytest
from pydantic import BaseModel

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
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


def _registry_with_tool() -> ToolRegistry:
    registry = ToolRegistry()
    contract = tools_agent_make_contract(TOOL_ID, _In, _Out)
    registry.register(contract, _EchoHandler())
    return registry


@pytest.mark.asyncio
async def test_catalog_declarative_invoker_routes_through_catalog() -> None:
    registry = _registry_with_tool()
    invoker = build_catalog_declarative_invoker_from_registry(registry)
    invoker.bind_run(
        run_id=mint_run_id(),
        task_id=mint_task_id(),
        agent_id="agent-a",
        tenant_id="tenant-1",
    )
    result = await invoker.invoke(
        tool_id=TOOL_ID,
        args={"value": 4},
        idempotency_key="key-1",
    )
    assert result.status == "success"
    assert result.output == {"result": 5}


def test_catalog_declarative_invoker_builds_real_runtime_context() -> None:
    registry = _registry_with_tool()
    invoker = build_catalog_declarative_invoker_from_registry(registry)
    invoker.bind_run(
        run_id=mint_run_id(),
        task_id=mint_task_id(),
        agent_id="agent-a",
        tenant_id="tenant-1",
    )
    state = invoker._runtime_state()  # noqa: SLF001 — wiring verification
    assert isinstance(state.context.session_manager, SessionManager)
    assert isinstance(state.context.config.llm_adapter, LLMAdapter)


@pytest.mark.asyncio
async def test_catalog_declarative_invoker_bind_run_updates_scope() -> None:
    registry = _registry_with_tool()
    invoker = CatalogDeclarativeToolInvoker(
        tool_invoker=build_catalog_declarative_invoker_from_registry(registry).tool_invoker,
    )
    bound_run_id = mint_run_id()
    invoker.bind_run(
        run_id=bound_run_id,
        task_id=mint_task_id(),
        agent_id="agent-b",
        tenant_id="tenant-2",
    )
    assert invoker.binding.run_id == bound_run_id
    assert invoker.binding.agent_id == "agent-b"
