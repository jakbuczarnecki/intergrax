# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-20 — DeterministicChainPattern acceptance tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.patterns.deterministic_chain import DeterministicChainPattern
from intergrax.runtime.nexus.tools.tool_chain_spec import (
    USER_QUERY_SOURCE,
    ChainStep,
    FieldRef,
    ToolChainSpec,
)
from intergrax.runtime.nexus.tools.tool_invocation_pattern import pattern_for_mode
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, canonical_execution_identity_scope, canonical_run_id_for_tests

pytestmark = pytest.mark.unit


class _SearchIn(BaseModel):
    query: str = Field(min_length=1)


class _SearchOut(BaseModel):
    context_text: str


class _FetchIn(BaseModel):
    query: str = Field(min_length=1)


class _FetchOut(BaseModel):
    result: str


class _SearchHandler:
    def execute(self, request: ToolExecutionRequest[_SearchIn]) -> _SearchOut:
        return _SearchOut(context_text=f"ctx:{request.input.query}")


class _FetchHandler:
    def execute(self, request: ToolExecutionRequest[_FetchIn]) -> _FetchOut:
        return _FetchOut(result=f"got:{request.input.query}")


def _chain_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolContract(
            tool_id="search.tool",
            name="search.tool",
            description="search",
            input_schema=_SearchIn,
            output_schema=_SearchOut,
            error_mapping={},
            side_effects=False,
            category="search",
        ),
        _SearchHandler(),
    )
    registry.register(
        ToolContract(
            tool_id="fetch.tool",
            name="fetch.tool",
            description="fetch",
            input_schema=_FetchIn,
            output_schema=_FetchOut,
            error_mapping={},
            side_effects=False,
            category="fetch",
        ),
        _FetchHandler(),
    )
    return registry


def _runtime_state(*, chain: ToolChainSpec, message: str) -> RuntimeState:
    from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
    from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor

    registry = _chain_registry()
    run_id = canonical_run_id_for_tests("run-chain-1")
    task_id = TaskId(f"task_{run_id[4:]}")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        tool_invocation_mode=ToolInvocationMode.DETERMINISTIC_CHAIN,
        tool_chain_spec=chain,
    )
    config.tool_invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="test-agent",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message=message,
            task_id=task_id,
            run_id=run_id,
        ),
        run_id=run_id,
    )


def test_pattern_for_mode_returns_deterministic_chain() -> None:
    pattern = pattern_for_mode(ToolInvocationMode.DETERMINISTIC_CHAIN)
    assert isinstance(pattern, DeterministicChainPattern)
    assert pattern.pattern_id == "deterministic_chain"


def test_deterministic_chain_maps_output_to_next_input() -> None:
    chain = ToolChainSpec(
        steps=[
            ChainStep(
                tool_id="search.tool",
                input_mappings={"query": USER_QUERY_SOURCE},
            ),
            ChainStep(
                tool_id="fetch.tool",
                input_mappings={"query": FieldRef(step=0, field="context_text")},
            ),
        ]
    )
    state = _runtime_state(chain=chain, message="widgets")
    invoker = state.context.config.tool_invoker
    assert invoker is not None
    pattern = DeterministicChainPattern()
    with canonical_execution_identity_scope(state.run_id):
        result = pattern.execute(
            state=state,
            invoker=invoker,
            planner=MagicMock(),
            plan=None,
            allowed_tool_ids=None,
            max_iterations=1,
            planner_input="widgets",
        )
    assert len(result.tool_traces) == 2
    assert all(trace.success for trace in result.tool_traces)
    assert result.tool_traces[1].output_preview is not None
    assert "got:ctx:widgets" in result.tool_traces[1].output_preview
