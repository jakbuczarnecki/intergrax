# © Artur Czarnecki. All rights reserved.

"""Unit gate — RuntimeToolInvoker cross-tenant invocation context (EBH-7 ST-01)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from intergrax.contracts.tools.tool_invocation_wiring_resolution import ToolInvocationContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from pydantic import BaseModel
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    tools_agent_make_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    n: int = 0


class _Out(BaseModel):
    n: int = 0


class _Handler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        return _Out(n=1)


def test_runtime_tool_invoker_rejects_cross_tenant_invocation_context() -> None:
    registry = ToolRegistry()
    tool_id = "ebh7.unit.cross_tenant"
    registry.register(tools_agent_make_contract(tool_id, _In, _Out), _Handler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    run_id = canonical_run_id_for_tests("ebh7-invoker-tenant")
    state = RuntimeState(
        context=RuntimeContext(
            config=RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False),
            session_manager=build_in_memory_session_manager(),
            prompt_registry=MagicMock(),
            context_builder=MagicMock(),
            rag_prompt_builder=MagicMock(),
            websearch_executor=MagicMock(),
            websearch_prompt_builder=MagicMock(),
        ),
        request=RuntimeRequest(
            agent_id="a",
            user_id="u",
            session_id="s",
            tenant_id="tenant-a",
            message="m",
            task_id="task_00000000000000000000000000000001",
            run_id=run_id,
        ),
        run_id=run_id,
        session=ChatSession(id="s", tenant_id="tenant-a", user_id="u"),
        tool_traces=[],
    )
    request = ToolExecutionRequest(
        run_id=str(run_id),
        step_id="step-1",
        tool_id=tool_id,
        input=_In(),
        invocation_context=ToolInvocationContext(
            run_id=str(run_id),
            step_id="step-1",
            tool_id=tool_id,
            tenant_id="tenant-b",
        ),
    )
    with (
        canonical_execution_identity_scope(run_id),
        pytest.raises(ToolScopeViolationError),
    ):
        invoker.invoke(state=state, agent_id="a", request=request)
