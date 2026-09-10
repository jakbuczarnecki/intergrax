# © Artur Czarnecki. All rights reserved.

"""Enterprise Scale & Resilience W1-A — root execution capacity admission lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityAdmissionTimeoutError,
    ExecutionCapacityExceededError,
    ExecutionCapacityOverloadMode,
    ExecutionCapacityPermit,
    ExecutionCapacityPolicy,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.local_execution_capacity_admission import (
    LocalExecutionCapacityAdmission,
    _LocalRootCapacityState,
)
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AUTHORITY = ParentExecutionAuthority.unrestricted_root()


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    label: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    label: str


class ConcurrencyObservingDelegate:
    def __init__(self, *, hold: asyncio.Event | None = None) -> None:
        self._hold = hold
        self.active = 0
        self.max_observed = 0
        self.enter_count = 0
        self._lock = asyncio.Lock()

    async def execute(self, request: ProbeRequest) -> ProbeResult:
        async with self._lock:
            self.enter_count += 1
            self.active += 1
            if self.active > self.max_observed:
                self.max_observed = self.active
        try:
            if self._hold is not None:
                await self._hold.wait()
            else:
                await asyncio.sleep(0.03)
            return ProbeResult(label=request.label)
        finally:
            async with self._lock:
                self.active -= 1


class RaisingDelegate:
    async def execute(self, request: ProbeRequest) -> ProbeResult:
        raise RuntimeError(f"boom:{request.label}")


class SelectiveRaisingDelegate:
    async def execute(self, request: ProbeRequest) -> ProbeResult:
        if request.label == "a":
            raise RuntimeError(f"boom:{request.label}")
        return ProbeResult(label=request.label)


def _root_context(*, execution_id=None) -> RootExecutionContext:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    exec_id = execution_id if execution_id is not None else mint_execution_id()
    return RootExecutionContext(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=exec_id,
        authority=_AUTHORITY,
        tenant_id="tenant-a",
        task_id=mint_task_id(),
    )


def _admission_request() -> ExecutionCapacityAdmissionRequest:
    return ExecutionCapacityAdmissionRequest(
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _runtime(
    delegate: (
        ConcurrencyObservingDelegate | SelectiveRaisingDelegate
    ),
    *,
    max_roots: int = 1,
    overload_mode: ExecutionCapacityOverloadMode = ExecutionCapacityOverloadMode.REJECT,
    wait_timeout_seconds: float | None = None,
) -> ExecutionRuntime[ProbeRequest, ProbeResult]:
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=max_roots,
        overload_mode=overload_mode,
        wait_timeout_seconds=wait_timeout_seconds,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    return ExecutionRuntime(
        delegate,
        execution_capacity_admission=admission,
    )


@pytest.mark.asyncio
async def test_bounded_root_concurrency() -> None:
    hold = asyncio.Event()
    delegate = ConcurrencyObservingDelegate(hold=hold)
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=3,
        overload_mode=ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=30.0,
    )
    runtime = ExecutionRuntime(
        delegate,
        execution_capacity_admission=LocalExecutionCapacityAdmission(policy),
    )
    contexts = [_root_context() for _ in range(20)]
    tasks = [
        asyncio.create_task(
            runtime.execute(ProbeRequest(label=str(index)), contexts[index]),
        )
        for index in range(20)
    ]
    await asyncio.sleep(0.05)
    assert delegate.max_observed <= 3
    hold.set()
    await asyncio.gather(*tasks)
    assert delegate.enter_count == 20


@pytest.mark.asyncio
async def test_release_on_success_allows_next_execution() -> None:
    delegate = ConcurrencyObservingDelegate()
    runtime = _runtime(delegate, max_roots=1)
    ctx_a = _root_context()
    ctx_b = _root_context()
    await runtime.execute(ProbeRequest(label="a"), ctx_a)
    await runtime.execute(ProbeRequest(label="b"), ctx_b)
    assert delegate.enter_count == 2


@pytest.mark.asyncio
async def test_release_on_exception_allows_next_execution() -> None:
    runtime = _runtime(SelectiveRaisingDelegate(), max_roots=1)
    ctx_a = _root_context()
    ctx_b = _root_context()
    with pytest.raises(RuntimeError, match="boom:a"):
        await runtime.execute(ProbeRequest(label="a"), ctx_a)
    await runtime.execute(ProbeRequest(label="b"), ctx_b)


@pytest.mark.asyncio
async def test_release_on_cancellation_allows_next_execution() -> None:
    hold = asyncio.Event()
    delegate = ConcurrencyObservingDelegate(hold=hold)
    runtime = _runtime(delegate, max_roots=1)
    ctx_a = _root_context()
    ctx_b = _root_context()
    task_a = asyncio.create_task(
        runtime.execute(ProbeRequest(label="a"), ctx_a),
    )
    while delegate.active < 1:
        await asyncio.sleep(0.001)
    task_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a
    hold.set()
    result = await runtime.execute(ProbeRequest(label="b"), ctx_b)
    assert result.label == "b"


@pytest.mark.asyncio
async def test_reject_overload_delegate_not_started() -> None:
    hold = asyncio.Event()
    delegate = ConcurrencyObservingDelegate(hold=hold)
    runtime = _runtime(delegate, max_roots=1)
    ctx_a = _root_context()
    ctx_b = _root_context()
    task_a = asyncio.create_task(
        runtime.execute(ProbeRequest(label="a"), ctx_a),
    )
    await asyncio.sleep(0.02)
    with pytest.raises(ExecutionCapacityExceededError):
        await runtime.execute(ProbeRequest(label="b"), ctx_b)
    assert delegate.enter_count == 1
    hold.set()
    await task_a


@pytest.mark.asyncio
async def test_wait_with_timeout_rejects_without_leak() -> None:
    hold = asyncio.Event()
    delegate = ConcurrencyObservingDelegate(hold=hold)
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=1,
        overload_mode=ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=0.05,
    )
    runtime = ExecutionRuntime(
        delegate,
        execution_capacity_admission=LocalExecutionCapacityAdmission(policy),
    )
    ctx_a = _root_context()
    ctx_b = _root_context()
    task_a = asyncio.create_task(
        runtime.execute(ProbeRequest(label="a"), ctx_a),
    )
    await asyncio.sleep(0.02)
    with pytest.raises(ExecutionCapacityAdmissionTimeoutError):
        await runtime.execute(ProbeRequest(label="b"), ctx_b)
    assert delegate.enter_count == 1
    hold.set()
    await task_a


def test_invalid_policy_values() -> None:
    with pytest.raises(ValidationError):
        ExecutionCapacityPolicy(max_concurrent_root_executions=0)
    with pytest.raises(ValidationError):
        ExecutionCapacityPolicy(
            max_concurrent_root_executions=2,
            overload_mode=ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT,
        )
    with pytest.raises(ValidationError):
        ExecutionCapacityPolicy(
            max_concurrent_root_executions=2,
            overload_mode=ExecutionCapacityOverloadMode.REJECT,
            wait_timeout_seconds=1.0,
        )


@pytest.mark.asyncio
async def test_local_permit_idempotent_release() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    permit = await admission.acquire(request)
    await permit.release()
    await permit.release()
    permit_b = await admission.acquire(request)
    await permit_b.release()


@pytest.mark.asyncio
async def test_reject_concurrent_acquire_capacity_one_atomic() -> None:
    """Regression: REJECT must atomically reserve or fail (no TOCTOU wait on semaphore)."""
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=1,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    contender_count = 32
    start_barrier = asyncio.Barrier(contender_count)
    permits: list[ExecutionCapacityPermit] = []
    errors: list[ExecutionCapacityExceededError] = []
    permits_lock = asyncio.Lock()

    async def contender() -> None:
        await start_barrier.wait()
        try:
            permit = await admission.acquire(request)
        except ExecutionCapacityExceededError as exc:
            async with permits_lock:
                errors.append(exc)
            return
        async with permits_lock:
            permits.append(permit)

    tasks = [asyncio.create_task(contender()) for _ in range(contender_count)]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    finally:
        for permit in permits:
            await permit.release()

    assert len(permits) == 1
    assert len(errors) == contender_count - 1


@pytest.mark.asyncio
async def test_reject_concurrent_acquire_capacity_three() -> None:
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=3,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    contender_count = 10
    start_barrier = asyncio.Barrier(contender_count)
    permits: list[ExecutionCapacityPermit] = []
    errors: list[ExecutionCapacityExceededError] = []
    permits_lock = asyncio.Lock()

    async def contender() -> None:
        await start_barrier.wait()
        try:
            permit = await admission.acquire(request)
        except ExecutionCapacityExceededError as exc:
            async with permits_lock:
                errors.append(exc)
            return
        async with permits_lock:
            permits.append(permit)

    tasks = [asyncio.create_task(contender()) for _ in range(contender_count)]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    finally:
        for permit in permits:
            await permit.release()

    assert len(permits) == 3
    assert len(errors) == contender_count - 3

    permit_extra = await admission.acquire(request)
    await permit_extra.release()


@pytest.mark.asyncio
async def test_reject_does_not_enqueue_waiter_when_saturated() -> None:
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=1,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    holder = await admission.acquire(request)
    reject_count = 16
    start_barrier = asyncio.Barrier(reject_count)

    async def reject_attempt() -> None:
        await start_barrier.wait()
        with pytest.raises(ExecutionCapacityExceededError):
            await admission.acquire(request)

    tasks = [asyncio.create_task(reject_attempt()) for _ in range(reject_count)]
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
    await holder.release()


@pytest.mark.asyncio
async def test_wait_with_timeout_acquires_after_release() -> None:
    hold = asyncio.Event()
    delegate = ConcurrencyObservingDelegate(hold=hold)
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=1,
        overload_mode=ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=5.0,
    )
    runtime = ExecutionRuntime(
        delegate,
        execution_capacity_admission=LocalExecutionCapacityAdmission(policy),
    )
    ctx_a = _root_context()
    ctx_b = _root_context()
    task_a = asyncio.create_task(
        runtime.execute(ProbeRequest(label="a"), ctx_a),
    )
    while delegate.active < 1:
        await asyncio.sleep(0.001)
    task_b = asyncio.create_task(
        runtime.execute(ProbeRequest(label="b"), ctx_b),
    )
    await asyncio.sleep(0.05)
    hold.set()
    result_b = await task_b
    await task_a
    assert result_b.label == "b"
    assert delegate.enter_count == 2


@pytest.mark.asyncio
async def test_local_permit_concurrent_double_release() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    permit = await admission.acquire(request)
    await asyncio.gather(permit.release(), permit.release())
    replacement = await admission.acquire(request)
    await replacement.release()


@pytest.mark.asyncio
async def test_local_permit_release_cancel_during_lifecycle_frees_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during permit.release() must not leak the slot."""
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    permit = await admission.acquire(request)
    release_entered = asyncio.Event()
    original_release_one = _LocalRootCapacityState.release_one

    async def gated_release_one(self: _LocalRootCapacityState) -> None:
        release_entered.set()
        await asyncio.sleep(0)
        await original_release_one(self)

    monkeypatch.setattr(_LocalRootCapacityState, "release_one", gated_release_one)
    release_task = asyncio.create_task(permit.release())
    await release_entered.wait()
    release_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release_task
    replacement = await admission.acquire(request)
    await replacement.release()


@pytest.mark.asyncio
async def test_local_permit_concurrent_release_one_cancelled_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    permit = await admission.acquire(request)
    release_started = asyncio.Event()
    release_unblock = asyncio.Event()
    original_release_one = _LocalRootCapacityState.release_one

    async def slow_release_one(self: _LocalRootCapacityState) -> None:
        release_started.set()
        await release_unblock.wait()
        await original_release_one(self)

    monkeypatch.setattr(_LocalRootCapacityState, "release_one", slow_release_one)
    first = asyncio.create_task(permit.release())
    await release_started.wait()
    second = asyncio.create_task(permit.release())
    await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release_unblock.set()
    await second
    replacement = await admission.acquire(request)
    await replacement.release()


@pytest.mark.asyncio
async def test_capacity_one_never_two_live_permits_after_release_races() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    request = _admission_request()
    permit = await admission.acquire(request)
    await asyncio.gather(
        permit.release(),
        permit.release(),
        permit.release(),
    )
    holder_a = await admission.acquire(request)
    with pytest.raises(ExecutionCapacityExceededError):
        await admission.acquire(request)
    await holder_a.release()
    holder_b = await admission.acquire(request)
    await holder_b.release()
