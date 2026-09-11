# © Artur Czarnecki. All rights reserved.

"""W4-A — cancellation lifecycle qualification matrix."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_capacity_admission import ExecutionCapacityPolicy
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.resilience import execute_with_resilience, reset_provider_resilience
from intergrax.runtime.cancellation.coordinator import (
    CooperativeCancellationAbort,
    cooperative_delay_seconds,
)
from intergrax.runtime.execution.concurrent_execution_work import execute_concurrent_execution_work
from intergrax.runtime.execution.local_execution_capacity_admission import (
    LocalExecutionCapacityAdmission,
)
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext
from intergrax.runtime.nexus.policies.policy_enforcer import PolicyEnforcer, TransientOperationError
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind, RuntimePolicies
from tests.unit.runtime.execution.test_concurrent_execution_work import (
    BarrierWorkPort,
    _request,
    _wait_until_started,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _EchoDelegate:
    async def execute(self, request: str) -> str:
        return request


_AUTHORITY = ParentExecutionAuthority.unrestricted_root()


def test_cooperative_delay_aborts_when_requested() -> None:
    gate = threading.Event()
    captured: list[BaseException] = []

    def _run() -> None:
        try:
            cooperative_delay_seconds(5.0, should_abort=gate.is_set)
        except BaseException as exc:
            captured.append(exc)

    worker = threading.Thread(target=_run)
    worker.start()
    time.sleep(0.05)
    gate.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(captured) == 1
    assert isinstance(captured[0], CooperativeCancellationAbort)


def test_provider_retry_cancel_during_backoff() -> None:
    reset_provider_resilience("openai")
    cfg = LLMCallConfig(max_retries=3, retry_backoff_sec=2.0)
    calls = {"n": 0}
    abort = threading.Event()

    class Transient(RuntimeError):
        status_code = 503

    def _fn() -> str:
        calls["n"] += 1
        raise Transient("down")

    def _run_resilience() -> None:
        try:
            execute_with_resilience(
                _fn,
                provider="openai",
                config=cfg,
                retry_fn=lambda f: f(),
                should_abort=abort.is_set,
            )
        except CooperativeCancellationAbort:
            return

    worker = threading.Thread(target=_run_resilience)
    worker.start()
    time.sleep(0.05)
    abort.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert calls["n"] == 1
    reset_provider_resilience("openai")


@pytest.mark.asyncio
async def test_policy_enforcer_cancel_during_backoff() -> None:
    from intergrax.runtime.nexus.policies.runtime_policies import RetryPolicy

    policies = RuntimePolicies(retry=RetryPolicy(max_attempts=3, backoff_seconds=2.0))
    enforcer = PolicyEnforcer(policies)

    async def _fail() -> str:
        raise TransientOperationError("transient")

    class _TraceOnlyState:
        def trace_event(self, **_kwargs: object) -> None:
            return

    state = _TraceOnlyState()

    task = asyncio.create_task(
        enforcer.execute(
            kind=ExecutionKind.LLM,
            op_name="w4",
            fn=_fail,
            state=state,
        ),
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_parent_execution_cancel_releases_capacity() -> None:
    hold = asyncio.Event()

    class _ObservingHoldDelegate:
        active = 0

        async def execute(self, request: str) -> str:
            self.active += 1
            await hold.wait()
            return request

    delegate = _ObservingHoldDelegate()
    admission = LocalExecutionCapacityAdmission(
        ExecutionCapacityPolicy(max_concurrent_root_executions=1),
    )
    runtime = ExecutionRuntime(delegate=delegate, execution_capacity_admission=admission)
    ctx_a = RootExecutionContext(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=_AUTHORITY,
        tenant_id="t",
        task_id=mint_task_id(),
    )
    ctx_b = RootExecutionContext(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        authority=_AUTHORITY,
        tenant_id="t",
        task_id=mint_task_id(),
    )
    blocker = asyncio.create_task(runtime.execute("hold", ctx_a))
    while delegate.active < 1:
        await asyncio.sleep(0.001)
    blocker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocker
    hold.set()
    echo_runtime = ExecutionRuntime(delegate=_EchoDelegate(), execution_capacity_admission=admission)
    assert await echo_runtime.execute("next", ctx_b) == "next"


@pytest.mark.asyncio
async def test_fan_out_100_children_cancel_stops_workers() -> None:
    release = asyncio.Event()
    started: list[str] = []
    port = BarrierWorkPort(release=release, started=started)
    labels = tuple(f"c{i}" for i in range(100))
    policy = ConcurrentExecutionWorkPolicy(max_concurrency=8)
    task = asyncio.create_task(
        execute_concurrent_execution_work(
            port,
            tuple(_request(label) for label in labels),
            policy=policy,
        ),
    )
    await _wait_until_started(started, frozenset(labels[:8]))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)
    assert len(started) < 100


def test_recovery_cancel_matrix_delegated_to_w3_c4() -> None:
    """Recovery cancel + permit release: ``test_decision_durable_recovery_w3_c4.py``."""
    from intergrax.contracts.recovery_admission import RecoveryKind

    assert RecoveryKind.DECISION_DURABLE.value == "DECISION_DURABLE"
