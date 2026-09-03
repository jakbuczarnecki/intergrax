# © Artur Czarnecki. All rights reserved.

"""Grant candidate resolution through execute_planned_tool_calls (ADR A–P)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from unittest.mock import MagicMock

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlCandidateStatus,
    DeclarativeHitlGrantCandidateMismatch,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_loop import execute_planned_tool_calls
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.core.tool_plan import PlannedToolCall
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager, canonical_execution_identity_scope, canonical_run_id_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TOOL_A = "grant.read.a"
_TOOL_B = "grant.read.b"


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    value: int = 0


class _CountingHandler:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        self.calls += 1
        return _Out(value=request.input.value)


def _registry(handler_a: _CountingHandler, handler_b: _CountingHandler) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id, handler in ((_TOOL_A, handler_a), (_TOOL_B, handler_b)):
        registry.register(
            ToolContract(
                tool_id=tool_id,
                name=tool_id,
                description="grant probe",
                input_schema=_In,
                output_schema=_Out,
                side_effects=False,
                error_mapping={},
                risk_level=ToolRiskLevel.LOW,
            ),
            handler,
        )
    return registry


def _state(run_id: str = "run-grant") -> RuntimeState:
    canonical_run_id = canonical_run_id_for_tests(run_id)
    canonical_task_id = TaskId(f"task_{canonical_run_id[4:]}")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        max_parallel_tool_calls=3,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message="grant probe",
            task_id=canonical_task_id,
            run_id=canonical_run_id,
        ),
        run_id=canonical_run_id,
    )


def _grant(**overrides: object) -> DeclarativeHitlApprovalGrant:
    canonical_run_id = canonical_run_id_for_tests("run-grant")
    canonical_task_id = TaskId(f"task_{canonical_run_id[4:]}")
    base = {
        "grant_id": "grant-1",
        "invocation_scope_id": "dhr_scope",
        "task_id": canonical_task_id,
        "run_id": canonical_run_id,
        "step_id": "step-a",
        "tool_id": _TOOL_A,
        "agent_id": "agent-1",
        "idempotency_key": None,
        "matched_rule_ids": ("rule-1",),
        "human_request_id": "hr-1",
        "policy_provenance_digest": None,
        "pause_id": "pause-1",
        "approved_at": "2026-08-14T00:00:00+00:00",
    }
    base.update(overrides)
    return DeclarativeHitlApprovalGrant(**base)


def test_ambiguous_matching_candidates_fail_closed_no_handler_calls() -> None:
    handler_a = _CountingHandler()
    handler_b = _CountingHandler()
    registry = _registry(handler_a, handler_b)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = _state()
    state.declarative_hitl_grant = _grant()
    calls = [
        PlannedToolCall(step_id="step-a", tool_id=_TOOL_A, input=_In(value=1)),
        PlannedToolCall(step_id="step-a", tool_id=_TOOL_A, input=_In(value=2)),
    ]
    with pytest.raises(DeclarativeHitlGrantCandidateMismatch) as exc_info:
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="ambig",
            max_parallel_read_only=3,
        )
    assert exc_info.value.status is DeclarativeHitlCandidateStatus.AMBIGUOUS
    assert handler_a.calls == 0
    assert handler_b.calls == 0


def test_no_matching_candidates_fail_closed_no_handler_calls() -> None:
    handler_a = _CountingHandler()
    handler_b = _CountingHandler()
    registry = _registry(handler_a, handler_b)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = _state()
    state.declarative_hitl_grant = _grant()
    calls = [
        PlannedToolCall(step_id="step-b", tool_id=_TOOL_B, input=_In(value=1)),
    ]
    with pytest.raises(DeclarativeHitlGrantCandidateMismatch) as exc_info:
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="nomatch",
            max_parallel_read_only=1,
        )
    assert exc_info.value.status is DeclarativeHitlCandidateStatus.NO_MATCH
    assert handler_a.calls == 0
    assert handler_b.calls == 0


@pytest.mark.skip(
    reason="EXECUTION ENGINE DEPENDENCY: tool_loop requires active execution budget binding (UE-8B)",
)
def test_unique_candidate_assigns_scope_only_to_target_and_executes_once() -> None:
    handler_a = _CountingHandler()
    handler_b = _CountingHandler()
    registry = _registry(handler_a, handler_b)
    invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    state = _state()
    state.declarative_hitl_grant = _grant()
    calls = [
        PlannedToolCall(step_id="step-a", tool_id=_TOOL_A, input=_In(value=1)),
        PlannedToolCall(step_id="step-b", tool_id=_TOOL_B, input=_In(value=2)),
    ]
    with canonical_execution_identity_scope(state.run_id):
        outcomes = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="unique",
            max_parallel_read_only=3,
        )
    assert handler_a.calls == 1
    assert handler_b.calls == 1
    assert len(outcomes) == 2
