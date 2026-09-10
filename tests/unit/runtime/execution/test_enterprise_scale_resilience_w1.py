# © Artur Czarnecki. All rights reserved.

"""Enterprise Scale & Resilience W1 — bounded concurrent work and global deadline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from intergrax.contracts.concurrent_execution_work import (
    MAX_CONCURRENT_EXECUTION_WORK,
    ConcurrentExecutionWorkPolicy,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    bind_root_execution_budget,
    peek_active_execution_global_deadline_monotonic,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.execution.attempt_lifecycle import AttemptLifecycleService, InMemoryAttemptLifecycleStore
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService, InMemoryExecutionTerminalStore
from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.orchestration.graph_runner import NexusGraphRunner
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class WorkResult:
    value: str


def _request(label: str) -> ExecutionRequest[str, WorkResult]:
    return ExecutionRequest(input=label, output_type=WorkResult)


class ConcurrencyObservingPort(ExecutionWorkPort[str, WorkResult, WorkResult]):
    def __init__(self) -> None:
        self._active = 0
        self.max_observed = 0
        self._lock = asyncio.Lock()

    async def execute(
        self,
        request: ExecutionRequest[str, WorkResult],
    ) -> WorkResult:
        async with self._lock:
            self._active += 1
            if self._active > self.max_observed:
                self.max_observed = self._active
        try:
            await asyncio.sleep(0.02)
            return WorkResult(value=request.input)
        finally:
            async with self._lock:
                self._active -= 1


class SlowBarrierPort(ExecutionWorkPort[str, WorkResult, WorkResult]):
    def __init__(self, *, release: asyncio.Event, started: list[str]) -> None:
        self._release = release
        self._started = started

    async def execute(
        self,
        request: ExecutionRequest[str, WorkResult],
    ) -> WorkResult:
        self._started.append(request.input)
        await self._release.wait()
        return WorkResult(value=request.input)


@pytest.mark.asyncio
async def test_bounded_concurrent_work_observed_max_concurrency() -> None:
    port = ConcurrencyObservingPort()
    labels = tuple(f"item-{index}" for index in range(100))
    requests = tuple(_request(label) for label in labels)
    policy = ConcurrentExecutionWorkPolicy(max_concurrency=3)
    results = await execute_concurrent_execution_work(port, requests, policy=policy)
    assert port.max_observed <= 3
    assert tuple(result.value for result in results) == labels


@pytest.mark.asyncio
async def test_bounded_concurrent_work_preserves_order() -> None:
    port = ConcurrencyObservingPort()
    labels = ("a", "b", "c", "d", "e")
    requests = tuple(_request(label) for label in labels)
    policy = ConcurrentExecutionWorkPolicy(max_concurrency=2)
    results = await execute_concurrent_execution_work(port, requests, policy=policy)
    assert tuple(result.value for result in results) == labels


@pytest.mark.asyncio
async def test_bounded_concurrent_work_strict_failure_semantics() -> None:
    class FailPort(ExecutionWorkPort[str, WorkResult, WorkResult]):
        async def execute(
            self,
            request: ExecutionRequest[str, WorkResult],
        ) -> WorkResult:
            if request.input == "bad":
                raise RuntimeError("failed: bad")
            return WorkResult(value=request.input)

    port = FailPort()
    with pytest.raises(RuntimeError, match="failed: bad"):
        await execute_concurrent_execution_work(
            port,
            (_request("ok"), _request("bad"), _request("tail")),
            policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
        )


@pytest.mark.asyncio
async def test_bounded_concurrent_work_resilient_per_item_failures() -> None:
    class FailPort(ExecutionWorkPort[str, WorkResult, WorkResult]):
        async def execute(
            self,
            request: ExecutionRequest[str, WorkResult],
        ) -> WorkResult:
            if request.input == "bad":
                raise RuntimeError("failed: bad")
            return WorkResult(value=request.input)

    port = FailPort()
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_request("ok"), _request("bad"), _request("tail")),
        policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[2].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED


@pytest.mark.asyncio
async def test_bounded_concurrent_work_cancellation_propagates() -> None:
    release = asyncio.Event()
    started: list[str] = []
    port = SlowBarrierPort(release=release, started=started)
    task = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            port,
            tuple(_request(f"item-{index}") for index in range(8)),
            policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
        ),
    )
    for _ in range(50):
        if started:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_concurrent_execution_work_policy_invalid_max_concurrency_fail_closed() -> None:
    with pytest.raises(ValidationError):
        ConcurrentExecutionWorkPolicy(max_concurrency=0)
    with pytest.raises(ValidationError):
        ConcurrentExecutionWorkPolicy(max_concurrency=-1)
    with pytest.raises(ValidationError):
        ConcurrentExecutionWorkPolicy(max_concurrency=MAX_CONCURRENT_EXECUTION_WORK + 1)


def test_root_execution_global_deadline_fixed_at_bind() -> None:
    ledger = create_execution_budget_ledger(RunBudget(max_wall_time_seconds=30.0))
    execution_id = mint_execution_id()
    before = time.monotonic()
    token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=ledger,
        run_budget=RunBudget(max_wall_time_seconds=30.0),
    )
    try:
        deadline = peek_active_execution_global_deadline_monotonic()
        assert deadline is not None
        assert deadline >= before + 30.0
        assert deadline <= time.monotonic() + 30.0
    finally:
        reset_active_execution_budget(token)


def _build_graph_runner(lifecycle_service: AttemptLifecycleService) -> NexusGraphRunner:
    return NexusGraphRunner(
        registry=MagicMock(),
        graph_executor=MagicMock(),
        validation_engine=MagicMock(),
        composer=FinalResponseComposer(),
        hitl=MagicMock(),
        events=MagicMock(),
        finish_task=AsyncMock(),
        finalize_trace=AsyncMock(),
        maybe_checkpoint=AsyncMock(),
        attempt_lifecycle=lifecycle_service,
        execution_terminal=ExecutionTerminalService(InMemoryExecutionTerminalStore()),
    )


def test_graph_runner_retry_rejected_when_global_deadline_exceeded() -> None:
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_a1,
        execution_id=execution_id,
    )
    ledger = create_execution_budget_ledger(None)
    budget_token = bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
            global_deadline_monotonic=time.monotonic() - 1.0,
        ),
    )
    task = Task(task_id=mint_task_id(), tenant_id="tenant-a", user_id="user", message="hello")
    lifecycle_service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    lifecycle_service.record_initial_attempt(
        tenant_id=task.tenant_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    runner = _build_graph_runner(lifecycle_service)
    try:
        new_attempt_id = runner._transition_attempt_for_retry(
            task,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
        )
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)

    assert new_attempt_id is None


def test_graph_runner_propagates_active_global_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[float | None] = []

    class _CapturingRetryService:
        def compute_backoff(self, **kwargs: object) -> float:
            return 1.0

        def transition_for_retry(self, **kwargs: object) -> None:
            request = kwargs["request"]
            captured.append(request.global_deadline_monotonic)
            return None

    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_a1,
        execution_id=execution_id,
    )
    ledger = create_execution_budget_ledger(None)
    expected_deadline = time.monotonic() + 120.0
    budget_token = bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
            global_deadline_monotonic=expected_deadline,
        ),
    )
    task = Task(task_id=mint_task_id(), tenant_id="tenant-a", user_id="user", message="hello")
    runner = _build_graph_runner(AttemptLifecycleService(InMemoryAttemptLifecycleStore()))
    monkeypatch.setattr(runner, "_attempt_retry_service", lambda: _CapturingRetryService())
    try:
        runner._transition_attempt_for_retry(
            task,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
        )
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)

    assert captured == [expected_deadline]
