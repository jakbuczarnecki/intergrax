# © Artur Czarnecki. All rights reserved.

"""Canonical orchestration topology submission E2E proof (Execution/Nexus)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_parent_execution_id,
    require_active_execution_id,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSchedulingPolicy,
    OrchestrationSlot,
    OrchestrationSlotExecutionError,
    OrchestrationSlotId,
    OrchestrationSlotStatus,
    OrchestrationTopology,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_host_task,
    build_orchestration_topology_submission_port,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.asyncio]


async def _execute_with_root_budget(
    delegate: object,
    *,
    root_execution_id: str,
) -> object:
    budget_token = None
    if peek_active_execution_budget() is None:
        budget_token = bind_root_execution_budget(
            execution_id=root_execution_id,
            ledger=create_execution_budget_ledger(None),
        )
    try:
        return await ExecutionBoundary(
            delegate,
            identity=ExecutionIdentityBinding(
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=root_execution_id,
            ),
            authority=ParentExecutionAuthority.unknown(),
        ).execute(None)
    finally:
        if budget_token is not None:
            reset_active_execution_budget(budget_token)


@dataclass(frozen=True, slots=True)
class SquareWork:
    value: int


@dataclass(frozen=True, slots=True)
class FailWork:
    reason: str


ProofPayload = SquareWork | FailWork


class ProofTopologySlotExecutor:
    __slots__ = ("child_execution_ids",)

    def __init__(self) -> None:
        self.child_execution_ids: list[str] = []

    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: ProofPayload,
    ) -> int:
        del slot_id
        child_id = require_active_execution_id()
        parent_id = peek_active_parent_execution_id()
        assert parent_id is not None
        assert child_id != parent_id
        self.child_execution_ids.append(child_id)
        if isinstance(payload, FailWork):
            raise OrchestrationSlotExecutionError(
                code="expected_failure",
                message=payload.reason,
            )
        return payload.value * payload.value


@pytest.mark.asyncio
async def test_canonical_orchestration_topology_submission_proof() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    assert submission_port is not None
    assert nexus_loop.graph_executor is nexus_loop.graph_executor

    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    topology = OrchestrationTopology(
        slots=(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("a"),
                payload=SquareWork(2),
            ),
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("b"),
                payload=SquareWork(3),
            ),
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("c"),
                payload=FailWork("x"),
            ),
        )
    )
    slot_executor = ProofTopologySlotExecutor()
    root_execution_id = mint_execution_id()

    class _SubmissionDelegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await submission_port.submit(
                    topology,
                    OrchestrationSchedulingPolicy(max_concurrency=2),
                    slot_executor,
                )
            finally:
                governed.reset(token)

    result = await _execute_with_root_budget(
        _SubmissionDelegate(),
        root_execution_id=root_execution_id,
    )

    assert [outcome.slot_id for outcome in result.outcomes] == [
        OrchestrationSlotId("a"),
        OrchestrationSlotId("b"),
        OrchestrationSlotId("c"),
    ]
    assert result.outcomes[0].status is OrchestrationSlotStatus.SUCCESS
    assert result.outcomes[0].result == 4
    assert result.outcomes[1].status is OrchestrationSlotStatus.SUCCESS
    assert result.outcomes[1].result == 9
    assert result.outcomes[2].status is OrchestrationSlotStatus.FAILURE
    assert result.outcomes[2].failure is not None
    assert result.outcomes[2].failure.code == "expected_failure"
    assert result.outcomes[2].failure.message == "x"
    assert len(slot_executor.child_execution_ids) == 3
    assert root_execution_id not in slot_executor.child_execution_ids


@pytest.mark.asyncio
async def test_orchestration_topology_deterministic_fan_in_order() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    release_events = {
        OrchestrationSlotId("a"): asyncio.Event(),
        OrchestrationSlotId("b"): asyncio.Event(),
        OrchestrationSlotId("c"): asyncio.Event(),
    }

    class OrderedSlotExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: SquareWork,
        ) -> int:
            await release_events[slot_id].wait()
            return payload.value

    topology = OrchestrationTopology(
        slots=(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("a"),
                payload=SquareWork(1),
            ),
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("b"),
                payload=SquareWork(2),
            ),
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("c"),
                payload=SquareWork(3),
            ),
        )
    )

    class _SubmissionDelegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await submission_port.submit(
                    topology,
                    OrchestrationSchedulingPolicy(max_concurrency=3),
                    OrderedSlotExecutor(),
                )
            finally:
                governed.reset(token)

    async def _release_in_completion_order() -> None:
        release_events[OrchestrationSlotId("c")].set()
        await asyncio.sleep(0.01)
        release_events[OrchestrationSlotId("a")].set()
        await asyncio.sleep(0.01)
        release_events[OrchestrationSlotId("b")].set()

    release_task = asyncio.create_task(_release_in_completion_order())
    try:
        result = await _execute_with_root_budget(
            _SubmissionDelegate(),
            root_execution_id=mint_execution_id(),
        )
    finally:
        await release_task

    assert [outcome.slot_id for outcome in result.outcomes] == [
        OrchestrationSlotId("a"),
        OrchestrationSlotId("b"),
        OrchestrationSlotId("c"),
    ]


@pytest.mark.asyncio
async def test_orchestration_topology_bounded_concurrency_enforced_by_graph_executor() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    active = 0
    peak = 0
    lock = asyncio.Lock()

    class ConcurrencyObservingExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: SquareWork,
        ) -> int:
            nonlocal active, peak
            del slot_id
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.03)
            async with lock:
                active -= 1
            return payload.value

    topology = OrchestrationTopology(
        slots=tuple(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId(f"s{i}"),
                payload=SquareWork(i),
            )
            for i in range(6)
        )
    )

    class _SubmissionDelegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await submission_port.submit(
                    topology,
                    OrchestrationSchedulingPolicy(max_concurrency=3),
                    ConcurrencyObservingExecutor(),
                )
            finally:
                governed.reset(token)

    await _execute_with_root_budget(
        _SubmissionDelegate(),
        root_execution_id=mint_execution_id(),
    )

    assert peak <= 3
    assert peak > 1


class _ConcurrencyTracker:
    __slots__ = ("active", "lock", "peak")

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = asyncio.Lock()

    async def observe(self) -> None:
        async with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.03)
        async with self.lock:
            self.active -= 1


@dataclass(frozen=True, slots=True)
class _TrackedWork:
    tracker: _ConcurrencyTracker
    value: int


async def _submit_topology(
    *,
    submission_port: object,
    host_task: object,
    topology: OrchestrationTopology[_TrackedWork],
    max_concurrency: int,
    slot_executor: object,
) -> object:
    class _SubmissionDelegate:
        async def execute(self, _request: object) -> object:
            governed = ActiveGovernedExecutionTask()
            token = governed.bind(host_task)
            try:
                return await submission_port.submit(
                    topology,
                    OrchestrationSchedulingPolicy(max_concurrency=max_concurrency),
                    slot_executor,
                )
            finally:
                governed.reset(token)

    return await _execute_with_root_budget(
        _SubmissionDelegate(),
        root_execution_id=mint_execution_id(),
    )


def _build_tracked_topology(
    *,
    prefix: str,
    slot_count: int,
    tracker: _ConcurrencyTracker,
) -> OrchestrationTopology[_TrackedWork]:
    return OrchestrationTopology(
        slots=tuple(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId(f"{prefix}{index}"),
                payload=_TrackedWork(tracker=tracker, value=index),
            )
            for index in range(slot_count)
        )
    )


class _TrackedSlotExecutor:
    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: _TrackedWork,
    ) -> int:
        del slot_id
        await payload.tracker.observe()
        return payload.value


@pytest.mark.asyncio
async def test_concurrent_topology_submissions_preserve_independent_scheduling_limits() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    tracker_a = _ConcurrencyTracker()
    tracker_b = _ConcurrencyTracker()
    topology_a = _build_tracked_topology(prefix="a", slot_count=6, tracker=tracker_a)
    topology_b = _build_tracked_topology(prefix="b", slot_count=8, tracker=tracker_b)
    start_gate = asyncio.Event()

    class _GatedTrackedSlotExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _TrackedWork,
        ) -> int:
            del slot_id
            await start_gate.wait()
            await payload.tracker.observe()
            return payload.value

    slot_executor = _GatedTrackedSlotExecutor()

    async def _run_a() -> object:
        return await _submit_topology(
            submission_port=submission_port,
            host_task=host_task,
            topology=topology_a,
            max_concurrency=2,
            slot_executor=slot_executor,
        )

    async def _run_b() -> object:
        return await _submit_topology(
            submission_port=submission_port,
            host_task=host_task,
            topology=topology_b,
            max_concurrency=4,
            slot_executor=slot_executor,
        )

    task_a = asyncio.create_task(_run_a())
    task_b = asyncio.create_task(_run_b())
    await asyncio.sleep(0.01)
    start_gate.set()
    result_a, result_b = await asyncio.gather(task_a, task_b)

    assert tracker_a.peak <= 2
    assert tracker_b.peak <= 4
    assert tracker_a.peak > 1
    assert tracker_b.peak > 1
    assert len(result_a.outcomes) == 6
    assert len(result_b.outcomes) == 8


@pytest.mark.asyncio
async def test_orchestration_topology_global_platform_cap_limits_submission() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry, max_parallel_nodes=3)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    tracker = _ConcurrencyTracker()
    topology = _build_tracked_topology(prefix="cap", slot_count=6, tracker=tracker)

    await _submit_topology(
        submission_port=submission_port,
        host_task=host_task,
        topology=topology,
        max_concurrency=10,
        slot_executor=_TrackedSlotExecutor(),
    )

    assert tracker.peak <= 3
    assert tracker.peak > 1


@pytest.mark.asyncio
async def test_orchestration_topology_submission_cap_limits_when_platform_higher() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry, max_parallel_nodes=10)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    tracker = _ConcurrencyTracker()
    topology = _build_tracked_topology(prefix="sub", slot_count=6, tracker=tracker)

    await _submit_topology(
        submission_port=submission_port,
        host_task=host_task,
        topology=topology,
        max_concurrency=3,
        slot_executor=_TrackedSlotExecutor(),
    )

    assert tracker.peak <= 3
    assert tracker.peak > 1


@dataclass(frozen=True, slots=True)
class _BugWork:
    marker: str


@pytest.mark.asyncio
async def test_orchestration_topology_programming_error_propagates_fail_fast() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    topology = OrchestrationTopology(
        slots=(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("a"),
                payload=SquareWork(2),
            ),
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("b"),
                payload=_BugWork("bug"),
            ),
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("c"),
                payload=SquareWork(4),
            ),
        )
    )

    class _BuggySlotExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: ProofPayload | _BugWork,
        ) -> int:
            del slot_id
            if isinstance(payload, _BugWork):
                raise TypeError("bug")
            if isinstance(payload, FailWork):
                raise OrchestrationSlotExecutionError(
                    code="expected_failure",
                    message=payload.reason,
                )
            return payload.value * payload.value

    with pytest.raises(TypeError, match="bug"):
        await _submit_topology(
            submission_port=submission_port,
            host_task=host_task,
            topology=topology,
            max_concurrency=3,
            slot_executor=_BuggySlotExecutor(),
        )


@pytest.mark.asyncio
async def test_orchestration_topology_success_with_none_result() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    host_task = build_orchestration_topology_host_task(
        tenant_id="tenant-proof",
        user_id="user-proof",
        task_id=mint_task_id(),
    )
    topology = OrchestrationTopology(
        slots=(
            OrchestrationSlot(
                slot_id=OrchestrationSlotId("a"),
                payload=SquareWork(0),
            ),
        )
    )

    class _NoneReturningExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: SquareWork,
        ) -> None:
            del slot_id, payload
            return None

    result = await _submit_topology(
        submission_port=submission_port,
        host_task=host_task,
        topology=topology,
        max_concurrency=1,
        slot_executor=_NoneReturningExecutor(),
    )

    assert result.outcomes[0].status is OrchestrationSlotStatus.SUCCESS
    assert result.outcomes[0].result is None
