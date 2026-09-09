# © Artur Czarnecki. All rights reserved.

"""NPSC-5B — final production fan-out / fan-in qualification proofs."""

from __future__ import annotations

import asyncio
import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutId,
    FanOutItemId,
    FanOutItemStatus,
    FanOutRequest,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationFailureCode,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLeaseId,
    TaskScopedAgentLeaseState,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_task_id,
    require_active_execution_id,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotStatus,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.fan_out_orchestration_adapter import (
    build_fan_out_orchestration_port,
    map_orchestration_outcome_to_fan_out,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_submission_port,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _ConcurrencyTracker,
    _DelayedOcrDelegate,
    _fan_out_item,
    build_fan_out_harness,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
    _root_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


def _build_shared_fan_out_service(
    harness,
    *,
    nexus_loop: NexusLoop,
):
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    coordination = _build_coordination_service(harness)
    orchestration = build_fan_out_orchestration_port(
        submission_port,
        coordination,
    )
    return BoundedMultiAgentFanOutService(orchestration=orchestration)


async def _run_fan_out_under_root(
    fan_out: BoundedMultiAgentFanOutService[OcrRequest, OcrResult],
    *,
    task_scope,
    items,
    fan_out_id: str,
    max_concurrency: int,
    start_gate: asyncio.Event | None = None,
):
    captured = []
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            if start_gate is not None:
                await start_gate.wait()
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=_UNLIMITED_LEDGER,
            )
            try:
                result = await fan_out.fan_out(
                    FanOutRequest(
                        fan_out_id=FanOutId(fan_out_id),
                        items=items,
                        max_concurrency=max_concurrency,
                    ),
                    principal=admin_test_principal(),
                )
            finally:
                reset_active_execution_budget(budget_token)
            captured.append(result)
            return OcrResult(text="root-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    assert captured
    return captured[0], root


@pytest.mark.asyncio
async def test_npsc5b_final_production_fanout_fanin_e2e_qualification() -> None:
    orchestration_child_ids: list[str] = []
    specialist_child_ids: list[str] = []
    tracker = _ConcurrencyTracker()

    class _LineageDelayedDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            await tracker.enter()
            try:
                specialist_child_ids.append(require_active_execution_id())
                if request.document_ref == "doc-d":
                    raise RuntimeError("expected specialist failure")
                await asyncio.sleep(0.03)
                return OcrResult(text=f"ocr:{request.document_ref}")
            finally:
                await tracker.leave()

    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_LineageDelayedDelegate(),
    )
    nexus_loop = NexusLoop(AgentRegistry())
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    coordination = _build_coordination_service(harness)

    class _TrackingSubmissionPort:
        def __init__(self, inner) -> None:
            self._inner = inner

        async def submit(self, topology, scheduling_policy, slot_executor):
            class _TrackingSlotExecutor:
                def __init__(self, wrapped) -> None:
                    self._wrapped = wrapped

                async def execute_slot(self, *, slot_id, payload):
                    orchestration_child_ids.append(require_active_execution_id())
                    return await self._wrapped.execute_slot(
                        slot_id=slot_id,
                        payload=payload,
                    )

            return await self._inner.submit(
                topology,
                scheduling_policy,
                _TrackingSlotExecutor(slot_executor),
            )

    fan_out = BoundedMultiAgentFanOutService(
        orchestration=build_fan_out_orchestration_port(
            _TrackingSubmissionPort(submission_port),
            coordination,
        ),
    )
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    items = tuple(
        _fan_out_item(
            item_id=f"item-{label}",
            task_scope=task_scope,
            coordination_id=f"coord-{label}",
            delegation_id=f"delegation-{label}",
            lease_id=f"lease-{label}",
            document_ref=f"doc-{label}",
        )
        for label in ("a", "b", "c", "d")
    )
    result, root = await _run_fan_out_under_root(
        fan_out,
        task_scope=task_scope,
        items=items,
        fan_out_id="npsc5b-final-e2e",
        max_concurrency=2,
    )

    assert tracker.peak <= 2
    assert tracker.peak > 1
    assert [item.item_id for item in result.items] == [
        FanOutItemId("item-a"),
        FanOutItemId("item-b"),
        FanOutItemId("item-c"),
        FanOutItemId("item-d"),
    ]
    successes = [item for item in result.items if item.status is FanOutItemStatus.SUCCESS]
    failures = [item for item in result.items if item.status is FanOutItemStatus.FAILURE]
    assert len(successes) == 3
    assert len(failures) == 1
    assert failures[0].item_id == FanOutItemId("item-d")
    assert failures[0].failure is not None
    assert (
        failures[0].failure.failure_code
        is CoordinationFailureCode.CHILD_EXECUTION_FAILED
    )
    for label in ("a", "b", "c", "d"):
        lease = harness.lease_store.get(TaskScopedAgentLeaseId(f"lease-{label}"))
        assert lease is not None
        assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED
    root_execution_id = root.execution_id
    assert root_execution_id not in orchestration_child_ids
    assert root_execution_id not in specialist_child_ids
    assert len(orchestration_child_ids) == 4
    assert len(specialist_child_ids) == 4
    assert orchestration_child_ids[0] != specialist_child_ids[0]
    for orch_id, spec_id in zip(
        orchestration_child_ids,
        specialist_child_ids,
        strict=True,
    ):
        assert orch_id != spec_id
        assert orch_id != root_execution_id
        assert spec_id != root_execution_id


@pytest.mark.asyncio
async def test_npsc5b_concurrent_fan_out_submissions_preserve_independent_limits() -> None:
    nexus_loop = NexusLoop(AgentRegistry())
    tracker_a = _ConcurrencyTracker()
    tracker_b = _ConcurrencyTracker()
    harness_a = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_DelayedOcrDelegate(tracker=tracker_a, delay_s=0.05),
    )
    harness_b = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_DelayedOcrDelegate(tracker=tracker_b, delay_s=0.05),
    )
    fan_out_a = _build_shared_fan_out_service(harness_a, nexus_loop=nexus_loop)
    fan_out_b = _build_shared_fan_out_service(harness_b, nexus_loop=nexus_loop)
    task_scope_a = mint_task_id()
    task_scope_b = mint_task_id()
    harness_a.task_scope_authority.task_scope_id = task_scope_a
    harness_b.task_scope_authority.task_scope_id = task_scope_b
    items_a = tuple(
        _fan_out_item(
            item_id=f"a-item-{index}",
            task_scope=task_scope_a,
            coordination_id=f"a-coord-{index}",
            delegation_id=f"a-delegation-{index}",
            lease_id=f"a-lease-{index}",
            document_ref=f"a-doc-{index}",
        )
        for index in range(6)
    )
    items_b = tuple(
        _fan_out_item(
            item_id=f"b-item-{index}",
            task_scope=task_scope_b,
            coordination_id=f"b-coord-{index}",
            delegation_id=f"b-delegation-{index}",
            lease_id=f"b-lease-{index}",
            document_ref=f"b-doc-{index}",
        )
        for index in range(8)
    )
    start_gate = asyncio.Event()

    task_a = asyncio.create_task(
        _run_fan_out_under_root(
            fan_out_a,
            task_scope=task_scope_a,
            items=items_a,
            fan_out_id="fan-out-a",
            max_concurrency=2,
            start_gate=start_gate,
        ),
    )
    task_b = asyncio.create_task(
        _run_fan_out_under_root(
            fan_out_b,
            task_scope=task_scope_b,
            items=items_b,
            fan_out_id="fan-out-b",
            max_concurrency=4,
            start_gate=start_gate,
        ),
    )
    await asyncio.sleep(0.01)
    start_gate.set()
    (result_a, _), (result_b, _) = await asyncio.gather(task_a, task_b)

    assert tracker_a.peak <= 2
    assert tracker_b.peak <= 4
    assert tracker_a.peak > 1
    assert tracker_b.peak > 1
    assert len(result_a.items) == 6
    assert len(result_b.items) == 8


@pytest.mark.asyncio
async def test_npsc5b_fan_out_platform_cap_limits_concurrency() -> None:
    tracker = _ConcurrencyTracker()
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_DelayedOcrDelegate(tracker=tracker, delay_s=0.05),
    )
    nexus_loop = NexusLoop(AgentRegistry(), max_parallel_nodes=2)
    fan_out = _build_shared_fan_out_service(harness, nexus_loop=nexus_loop)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    items = tuple(
        _fan_out_item(
            item_id=f"item-{index}",
            task_scope=task_scope,
            coordination_id=f"coord-{index}",
            delegation_id=f"delegation-{index}",
            lease_id=f"lease-{index}",
            document_ref=f"doc-{index}",
        )
        for index in range(6)
    )
    await _run_fan_out_under_root(
        fan_out,
        task_scope=task_scope,
        items=items,
        fan_out_id="fan-out-platform-cap",
        max_concurrency=5,
    )
    assert tracker.peak <= 2
    assert tracker.peak > 1


def test_npsc5b_skipped_orchestration_slot_maps_to_invalid_coordination() -> None:
    outcome = map_orchestration_outcome_to_fan_out(
        OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("item-a"),
            status=OrchestrationSlotStatus.SKIPPED,
            result=None,
            failure=None,
        ),
        expected_item_id=FanOutItemId("item-a"),
    )
    assert outcome.status is FanOutItemStatus.FAILURE
    assert outcome.failure is not None
    assert (
        outcome.failure.failure_code
        is CoordinationFailureCode.INVALID_COORDINATION
    )
