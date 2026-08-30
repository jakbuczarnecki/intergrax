# © Artur Czarnecki. All rights reserved.

"""UE-9AR1 — preserve one Run budget across background redelivery attempts."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedger,
    ExecutionBudgetLedgerFactory,
    create_execution_budget_ledger_factory,
)
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationContext,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
    ExecutionBudgetError,
)
from intergrax.runtime.execution.budget.persistence import (
    KvRunBudgetPersistence,
    RunBudgetPersistenceError,
    create_durable_run_budget_ledger_factory,
)
from intergrax.runtime.execution.budget.policy import DefaultSharedPoolBudgetPolicy
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-ue9ar1"
_LIMIT = RunBudget(max_total_tokens=100, max_tool_calls=100)


class _KV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str, key: str) -> bytes | None:
        with self._lock:
            return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        with self._lock:
            self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        with self._lock:
            self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        del ttl_seconds
        with self._lock:
            current = self._data.get((tenant_id, key))
            if expected is None and current is not None:
                return False
            if expected is not None and current != expected:
                return False
            self._data[(tenant_id, key)] = new_value
            return True


def _factory(kv: _KV | None = None) -> ExecutionBudgetLedgerFactory:
    return create_durable_run_budget_ledger_factory(
        KvRunBudgetPersistence(kv or _KV()),
        _LIMIT,
    )


def _consume_total_tokens(
    ledger: ExecutionBudgetLedger,
    *,
    root_execution_id: ExecutionId,
    amount: int,
) -> None:
    child_id = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=child_id,
        parent_execution_id=root_execution_id,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )
    ledger.consume_budget(child_id, BudgetUsageTotals(total_tokens=amount))
    ledger.release_child_budget(child_id)


def _open_ledger(
    factory: ExecutionBudgetLedgerFactory,
    *,
    run_id: RunId,
    attempt_id: AttemptId,
) -> ExecutionBudgetLedger:
    return factory.create_ledger(
        _LIMIT,
        tenant_id=_TENANT,
        run_id=run_id,
        attempt_id=attempt_id,
    )


def test_attempt_two_gets_remaining_budget_after_attempt_one_consumption() -> None:
    factory = _factory()
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    attempt_two = mint_attempt_id()
    root_one = mint_execution_id()

    ledger_one = _open_ledger(factory, run_id=run_id, attempt_id=attempt_one)
    _consume_total_tokens(ledger_one, root_execution_id=root_one, amount=70)

    ledger_two = _open_ledger(factory, run_id=run_id, attempt_id=attempt_two)
    assert ledger_two.snapshot_root_available().max_total_tokens == 30


def test_attempt_two_has_new_attempt_id_but_same_run_budget_state() -> None:
    factory = _factory()
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    attempt_two = mint_attempt_id()
    assert attempt_one != attempt_two

    ledger_one = _open_ledger(factory, run_id=run_id, attempt_id=attempt_one)
    _consume_total_tokens(ledger_one, root_execution_id=mint_execution_id(), amount=25)

    ledger_two = _open_ledger(factory, run_id=run_id, attempt_id=attempt_two)
    assert ledger_two.snapshot_root_available().max_total_tokens == 75


def test_attempt_two_new_root_execution_id_does_not_reset_budget() -> None:
    factory = _factory()
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    attempt_two = mint_attempt_id()

    ledger_one = _open_ledger(factory, run_id=run_id, attempt_id=attempt_one)
    _consume_total_tokens(ledger_one, root_execution_id=mint_execution_id(), amount=60)

    root_two = mint_execution_id()
    ledger_two = _open_ledger(factory, run_id=run_id, attempt_id=attempt_two)
    _consume_total_tokens(ledger_two, root_execution_id=root_two, amount=10)
    assert ledger_two.snapshot_root_available().max_total_tokens == 30


def test_three_attempts_accumulate_consumption_and_leave_ten_remaining() -> None:
    factory = _factory()
    run_id = mint_run_id()
    amounts = (40, 30, 20)
    for amount in amounts:
        ledger = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
        _consume_total_tokens(ledger, root_execution_id=mint_execution_id(), amount=amount)

    final_ledger = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
    assert final_ledger.snapshot_root_available().max_total_tokens == 10


def test_next_attempt_fails_when_requesting_more_than_remaining_budget() -> None:
    factory = _factory()
    run_id = mint_run_id()
    for amount in (40, 30, 20):
        ledger = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
        _consume_total_tokens(ledger, root_execution_id=mint_execution_id(), amount=amount)

    over_budget = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
    child_id = mint_execution_id()
    root_id = mint_execution_id()
    over_budget.grant_child_budget(
        execution_id=child_id,
        parent_execution_id=root_id,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )
    with pytest.raises(ExecutionBudgetError):
        over_budget.consume_budget(child_id, BudgetUsageTotals(total_tokens=20))


def test_different_runs_keep_independent_budgets() -> None:
    factory = _factory()
    run_a = mint_run_id()
    run_b = mint_run_id()

    ledger_a = _open_ledger(factory, run_id=run_a, attempt_id=mint_attempt_id())
    _consume_total_tokens(ledger_a, root_execution_id=mint_execution_id(), amount=80)

    ledger_b = _open_ledger(factory, run_id=run_b, attempt_id=mint_attempt_id())
    assert ledger_b.snapshot_root_available().max_total_tokens == 100


def test_parallel_workers_same_run_cannot_overspend() -> None:
    kv = _KV()
    factory = create_durable_run_budget_ledger_factory(KvRunBudgetPersistence(kv), _LIMIT)
    run_id = mint_run_id()
    successes: list[int] = []
    failures: list[Exception] = []
    barrier = threading.Barrier(2)

    def _worker(amount: int) -> None:
        barrier.wait()
        ledger = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
        child_id = mint_execution_id()
        root_id = mint_execution_id()
        try:
            ledger.grant_child_budget(
                execution_id=child_id,
                parent_execution_id=root_id,
                decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
            )
            ledger.consume_budget(child_id, BudgetUsageTotals(total_tokens=amount))
            ledger.release_child_budget(child_id)
        except ExecutionBudgetError as exc:
            failures.append(exc)
        else:
            successes.append(amount)

    threads = [
        threading.Thread(target=_worker, args=(70,)),
        threading.Thread(target=_worker, args=(70,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 1
    final_ledger = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
    assert final_ledger.snapshot_root_available().max_total_tokens == 30


def test_corrupt_persisted_budget_state_fails_closed() -> None:
    kv = _KV()
    run_id = mint_run_id()
    kv.set(_TENANT, f"run_budget_ledger:{run_id}", b"not-json")
    factory = create_durable_run_budget_ledger_factory(KvRunBudgetPersistence(kv), _LIMIT)

    with pytest.raises(RunBudgetPersistenceError):
        _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())


@pytest.mark.asyncio
async def test_normal_non_background_run_still_gets_fresh_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    factory = create_execution_budget_ledger_factory(RunBudget(max_total_tokens=42))
    run_budget = RunBudget(max_total_tokens=42)
    loop = NexusLoop(
        AgentRegistry(),
        execution_budget_ledger_factory=factory,
        run_budget=run_budget,
    )
    observed: list[int | None] = []

    async def _observe(task: Task) -> TaskResult:
        from intergrax.runtime.execution.active_execution_budget import require_active_execution_budget

        observed.append(
            require_active_execution_budget().ledger.snapshot_root_available().max_total_tokens
        )
        return TaskResult(task_id=task.task_id, run_id=mint_run_id(), state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _observe)
    runner = UnifiedTaskRunner(
        loop,
        execution_budget_ledger_factory=factory,
        run_budget=run_budget,
    )
    await runner.run_task(
        Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="fresh"),
    )
    assert observed == [42]


@pytest.mark.asyncio
async def test_local_retry_does_not_create_new_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

    inner_factory = create_execution_budget_ledger_factory(RunBudget(max_total_tokens=100))
    factory = MagicMock(spec=ExecutionBudgetLedgerFactory, wraps=inner_factory)
    run_budget = RunBudget(max_total_tokens=100)
    loop = NexusLoop(
        AgentRegistry(),
        execution_budget_ledger_factory=factory,
        run_budget=run_budget,
    )

    async def _noop(task: Task) -> TaskResult:
        return TaskResult(task_id=task.task_id, run_id=mint_run_id(), state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _noop)
    runner = UnifiedTaskRunner(
        loop,
        execution_budget_ledger_factory=factory,
        run_budget=run_budget,
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="once")
    await runner.run_task(task)
    await runner.run_task(task)
    assert factory.create_ledger.call_count == 2


def test_child_reservation_accounting_remains_correct_after_restore() -> None:
    factory = _factory()
    run_id = mint_run_id()
    root_id = mint_execution_id()
    reserved_id = mint_execution_id()
    policy = DefaultSharedPoolBudgetPolicy()
    reservation = RunBudget(max_tool_calls=10)

    ledger_one = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
    ledger_one.grant_child_budget(
        execution_id=reserved_id,
        parent_execution_id=root_id,
        decision=policy.resolve_child_budget(
            ChildBudgetAllocationContext(
                parent_execution_id=root_id,
                parent_allocation_mode=ExecutionBudgetAllocationMode.SHARED,
                parent_reservation_remaining=None,
                requested_budget=reservation,
            )
        ),
    )
    ledger_one.consume_budget(reserved_id, BudgetUsageTotals(tool_calls=4))

    ledger_two = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
    new_root = mint_execution_id()
    new_child = mint_execution_id()
    ledger_two.grant_child_budget(
        execution_id=new_child,
        parent_execution_id=new_root,
        decision=policy.resolve_child_budget(
            ChildBudgetAllocationContext(
                parent_execution_id=new_root,
                parent_allocation_mode=ExecutionBudgetAllocationMode.SHARED,
                parent_reservation_remaining=None,
                requested_budget=RunBudget(max_tool_calls=5),
            )
        ),
    )
    assert ledger_two.snapshot_root_available().max_tool_calls == 91


def test_crash_with_active_reserved_child_commits_consumed_on_redelivery() -> None:
    factory = _factory()
    run_id = mint_run_id()
    root_id = mint_execution_id()
    reserved_id = mint_execution_id()
    policy = DefaultSharedPoolBudgetPolicy()

    ledger_one = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
    ledger_one.grant_child_budget(
        execution_id=reserved_id,
        parent_execution_id=root_id,
        decision=policy.resolve_child_budget(
            ChildBudgetAllocationContext(
                parent_execution_id=root_id,
                parent_allocation_mode=ExecutionBudgetAllocationMode.SHARED,
                parent_reservation_remaining=None,
                requested_budget=RunBudget(max_total_tokens=30),
            )
        ),
    )
    ledger_one.consume_budget(reserved_id, BudgetUsageTotals(total_tokens=12))

    ledger_two = _open_ledger(factory, run_id=run_id, attempt_id=mint_attempt_id())
    assert ledger_two.snapshot_root_available().max_total_tokens == 88
