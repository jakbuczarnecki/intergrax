# © Artur Czarnecki. All rights reserved.

"""UE-8B2 — runtime consumption wired into canonical ExecutionBudgetLedger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.consumption import (
    consume_llm_call,
    consume_llm_token_usage,
    consume_planner_iteration,
    consume_rag_invocation,
    consume_tool_call,
    consume_wall_time_delta,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
    ExecutionBudgetError,
)
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.execution.result import ExecutionStatus
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.budget.budget_ticks import (
    record_planner_iteration_and_enforce,
    record_rag_invocation_and_enforce,
    record_tool_call_and_enforce,
)
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


def _bind_budget(
    ledger: object,
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> tuple[object, object, object]:
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    budget_token = bind_root_execution_budget(execution_id=execution_id, ledger=ledger)
    return identity_token, authority_token, budget_token


def _reset_budget(
  identity_token: object,
  authority_token: object,
  budget_token: object,
) -> None:
    reset_active_execution_budget(budget_token)
    reset_active_execution_authority(authority_token)
    reset_active_execution_identity(identity_token)


def test_tool_calls_limit_two_third_rejected() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=2))
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    state = build_runtime_state_for_tests(run_id=str(run_id))
    tokens = _bind_budget(ledger, run_id=run_id, attempt_id=attempt_id, execution_id=execution_id)
    try:
        consume_tool_call()
        consume_tool_call()
        with pytest.raises(ExecutionBudgetError):
            consume_tool_call()
    finally:
        _reset_budget(*tokens)


def test_rag_limit_one_second_rejected() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_rag_invocations=1))
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    state = build_runtime_state_for_tests(run_id=str(run_id))
    state.context.config.run_budget = RunBudget(max_rag_invocations=1)
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        record_rag_invocation_and_enforce(state)
        with pytest.raises(ExecutionBudgetError):
            record_rag_invocation_and_enforce(state)
    finally:
        _reset_budget(*tokens)


def test_planner_iteration_exact_delta() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_planner_iterations=2))
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    state = build_runtime_state_for_tests(run_id=str(run_id))
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        record_planner_iteration_and_enforce(state)
        record_planner_iteration_and_enforce(state)
        assert state.planner_iteration_count == 2
        with pytest.raises(ExecutionBudgetError):
            record_planner_iteration_and_enforce(state)
    finally:
        _reset_budget(*tokens)


class _CountingAdapter(LLMAdapter):
    provider = "test"
    model = "test"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ):
        self.calls += 1
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
        from intergrax.llm_adapters._shared.call_lifecycle import LLMCallLifecycle
        from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage

        lifecycle = LLMCallLifecycle.begin(self, run_id=run_id)
        lifecycle.mark_success()
        lifecycle.end(LLMTokenUsage())
        return build_adapter_response(content="ok")


def test_llm_calls_limit_two_third_rejected() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_llm_calls=2))
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    adapter = _CountingAdapter()
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        adapter.generate_messages([], run_id=str(run_id))
        adapter.generate_messages([], run_id=str(run_id))
        with pytest.raises(ExecutionBudgetError):
            adapter.generate_messages([], run_id=str(run_id))
    finally:
        _reset_budget(*tokens)


def test_token_accounting_once_per_call() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        consume_llm_call()
        consume_llm_token_usage(input_tokens=30, output_tokens=20, total_tokens=50)
        assert ledger.snapshot_root_available().max_total_tokens == 50
        consume_llm_call()
        consume_llm_token_usage(input_tokens=30, output_tokens=20, total_tokens=50)
        assert ledger.snapshot_root_available().max_total_tokens == 0
    finally:
        _reset_budget(*tokens)


def test_wall_time_delta_not_cumulative() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_wall_time_seconds=10.0))
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        consume_wall_time_delta(3.0)
        consume_wall_time_delta(10.0)
        assert ledger.snapshot_root_available().max_wall_time_seconds == pytest.approx(0.0)
        with pytest.raises(ExecutionBudgetError):
            consume_wall_time_delta(11.0)
    finally:
        _reset_budget(*tokens)


@dataclass(frozen=True)
class Echo:
    value: str


@dataclass(frozen=True)
class EchoOut:
    value: str


@pytest.mark.asyncio
async def test_reserved_child_usage_reduces_reservation() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    root_id = mint_execution_id()
    child_runner = ChildExecutionRunner[Echo, EchoOut](ledger=ledger)
    child_execution_id: ExecutionId | None = None
    remaining_tokens: int | None = None

    class ChildDelegate:
        async def execute(self, request: Echo) -> EchoOut:
            from intergrax.contracts.execution_identity import require_active_execution_id

            nonlocal child_execution_id, remaining_tokens
            child_execution_id = require_active_execution_id()
            consume_llm_token_usage(input_tokens=20, output_tokens=0, total_tokens=20)
            remaining_tokens = ledger.snapshot_reservation_remaining(
                child_execution_id,
            ).max_total_tokens
            return EchoOut(value=request.value)

    class RootDelegate:
        async def execute(self, request: Echo) -> EchoOut:
            return await child_runner.execute(
                request=request,
                delegate=ChildDelegate(),
                requested_budget=RunBudget(max_total_tokens=30),
            )

    run_id = mint_run_id()
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=root_id,
    )
    try:
        await ExecutionBoundary[Echo, EchoOut](
            RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=run_id,
                attempt_id=mint_attempt_id(),
                execution_id=root_id,
            ),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(Echo(value="child"))
    finally:
        _reset_budget(*tokens)

    assert child_execution_id is not None
    assert remaining_tokens == 10


@pytest.mark.asyncio
async def test_nested_child_charges_nested_execution_id() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=5))
    root_id = mint_execution_id()
    child_runner = ChildExecutionRunner[Echo, EchoOut](ledger=ledger)
    charged_ids: list[ExecutionId] = []

    class GrandchildDelegate:
        async def execute(self, request: Echo) -> EchoOut:
            from intergrax.contracts.execution_identity import require_active_execution_id

            charged_ids.append(require_active_execution_id())
            consume_tool_call()
            return EchoOut(value=request.value)

    class ChildDelegate:
        async def execute(self, request: Echo) -> EchoOut:
            return await child_runner.execute(request=request, delegate=GrandchildDelegate())

    class RootDelegate:
        async def execute(self, request: Echo) -> EchoOut:
            return await child_runner.execute(request=request, delegate=ChildDelegate())

    run_id = mint_run_id()
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=root_id,
    )
    try:
        await ExecutionBoundary[Echo, EchoOut](
            RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=run_id,
                attempt_id=mint_attempt_id(),
                execution_id=root_id,
            ),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(Echo(value="nested"))
    finally:
        _reset_budget(*tokens)

    assert len(charged_ids) == 1
    assert ledger.snapshot_root_available().max_tool_calls == 4


@pytest.mark.asyncio
async def test_direct_inference_usage_reaches_ledger() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_llm_calls=2, max_total_tokens=40))
    run_id = mint_run_id()
    execution_id = mint_execution_id()

    @dataclass(frozen=True, slots=True)
    class Out:
        x: int

    class StructuredAdapter(_CountingAdapter):
        def supports_structured_output(self) -> bool:
            return True

        def generate_structured(self, messages, output_model, **kwargs):
            from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
            from intergrax.llm_adapters._shared.call_lifecycle import LLMCallLifecycle
            from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
            from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage

            lifecycle = LLMCallLifecycle.begin(self, run_id=kwargs.get("run_id"))
            lifecycle.mark_success()
            usage = LLMTokenUsage.from_counts(input_tokens=10, output_tokens=5)
            lifecycle.end(usage)
            return LLMStructuredResult(
                parsed=Out(x=1),
                response=build_adapter_response(content="ok"),
            )

    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        request = ExecutionRequest(
            input=(ChatMessage(role="user", content="hi"),),
            output_type=Out,
        )
        result = await InferenceExecutor(StructuredAdapter()).execute(request)
        assert result.status is ExecutionStatus.COMPLETED
    finally:
        _reset_budget(*tokens)

    assert ledger.snapshot_root_available().max_llm_calls == 1
    assert ledger.snapshot_root_available().max_total_tokens == 25


def test_fail_closed_without_active_budget() -> None:
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(RuntimeError, match="active execution budget required"):
            consume_tool_call()
    finally:
        reset_active_execution_identity(identity_token)


def test_uaep_replan_consumes_ledger() -> None:
    from intergrax.runtime.execution.budget.consumption import consume_replan
    from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler

    ledger = create_execution_budget_ledger(RunBudget(max_replans=1))
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    tokens = _bind_budget(
        ledger,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    handler = ExecutionInterruptHandler(allow_dynamic_replan=True)
    try:
        resolution = handler.resolve_decision(
            AgentDecision(type=AgentDecisionType.MODIFY_PLAN, reason="replan"),
            task_id="task_test",
            run_id=str(run_id),
            agent_id="agent-1",
            context={"nexus_replan_boundary": True},
        )
        assert not resolution.should_fail
        consume_replan()
        with pytest.raises(ExecutionBudgetError):
            consume_replan()
    finally:
        _reset_budget(*tokens)


def test_different_runs_isolated() -> None:
    ledger_a = create_execution_budget_ledger(RunBudget(max_tool_calls=1))
    ledger_b = create_execution_budget_ledger(RunBudget(max_tool_calls=1))
    run_a = mint_run_id()
    run_b = mint_run_id()
    exec_a = mint_execution_id()
    exec_b = mint_execution_id()
    tokens_a = _bind_budget(ledger_a, run_id=run_a, attempt_id=mint_attempt_id(), execution_id=exec_a)
    consume_tool_call()
    _reset_budget(*tokens_a)
    tokens_b = _bind_budget(ledger_b, run_id=run_b, attempt_id=mint_attempt_id(), execution_id=exec_b)
    try:
        consume_tool_call()
        assert ledger_b.snapshot_root_available().max_tool_calls == 0
    finally:
        _reset_budget(*tokens_b)
    assert ledger_a.snapshot_root_available().max_tool_calls == 0
