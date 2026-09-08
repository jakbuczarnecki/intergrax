# © Artur Czarnecki. All rights reserved.

"""NPSC-5B — bounded multi-agent fan-out / fan-in contract tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskLifecyclePlan,
    DelegationId,
)
from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutId,
    FanOutItem,
    FanOutItemId,
    FanOutItemStatus,
    FanOutRequest,
    InvalidFanOutError,
    MAX_FAN_OUT_CONCURRENCY,
    validate_fan_out_id,
    validate_fan_out_item_id,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    ChildExecutionFailedError,
    CoordinationDelegation,
    CoordinationFailureCode,
    CoordinationId,
    CoordinationRequest,
    MultiAgentCoordinationService,
    NoEligibleSpecialistError,
)
from intergrax.agent_distribution.task_capability_resolution import (
    build_task_capability_resolution_request,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLeaseId,
    TaskScopedAgentLeaseState,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.delegated_subtask_child_port import (
    as_child_execution_port,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _APP,
    _ENV,
    _LEGAL_PACKAGE,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
    build_delegated_harness,
)
from tests.unit.agent_distribution.test_task_scoped_agents import _task_acquire_request
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
    _coordination_request,
    _root_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FanOutAcquisitionPlanFactory:
    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs
        self._harness = None

    def bind_harness(self, harness) -> None:
        self._harness = harness

    def build_acquisition_plan(
        self,
        *,
        delegation_id: DelegationId,
        task_scope_id,
        application_id: str,
        application_environment_id: str,
        lease_id: TaskScopedAgentLeaseId,
        selected_identity,
    ) -> DelegatedSubtaskLifecyclePlan:
        del delegation_id, application_id, application_environment_id
        prior_revision_id = None
        pointer_revision = 0
        if self._harness is not None:
            serving = self._harness.stack.service.inspect_serving(
                application_id=_APP,
                application_environment_id=_ENV,
            )
            prior_revision_id = serving.traffic_serving_revision_id
            pointer_revision = serving.serving_pointer_revision
        revision_id = f"rev-{lease_id}"
        return DelegatedSubtaskLifecyclePlan(
            acquisition_request=_task_acquire_request(
                str(lease_id),
                task_scope_id,
                revision_id,
                identity=selected_identity,
                prior_revision_id=prior_revision_id,
                pointer_revision=pointer_revision,
                **self._kwargs,
            ),
        )


def build_fan_out_harness(*, candidates, specialist_delegate=None):
    harness = build_delegated_harness(
        candidates=candidates,
        specialist_delegate=specialist_delegate,
    )
    factory = _FanOutAcquisitionPlanFactory()
    harness.service._acquisition_plan_factory = factory
    factory.bind_harness(harness)
    return harness


def _fan_out_item(
    *,
    item_id: str,
    task_scope,
    coordination_id: str,
    delegation_id: str,
    lease_id: str,
    document_ref: str = "doc-1",
) -> FanOutItem[OcrRequest]:
    return FanOutItem(
        item_id=FanOutItemId(item_id),
        request=_coordination_request(
            task_scope=task_scope,
            coordination_id=coordination_id,
            delegation_id=delegation_id,
            lease_id=lease_id,
        ),
        delegation=CoordinationDelegation(
            payload=OcrRequest(document_ref=document_ref),
        ),
    )


def _build_fan_out_service(harness) -> BoundedMultiAgentFanOutService[OcrRequest, OcrResult]:
    return BoundedMultiAgentFanOutService(
        coordination=_build_coordination_service(harness),
    )


async def _run_fan_out(
    harness,
    *,
    task_scope,
    items: tuple[FanOutItem[OcrRequest], ...],
    fan_out_id: str = "fan-out-1",
    max_concurrency: int = 2,
):
    fan_out = _build_fan_out_service(harness)
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()
    captured = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            result = await fan_out.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId(fan_out_id),
                    items=items,
                    max_concurrency=max_concurrency,
                ),
                principal=admin_test_principal(),
            )
            captured.append(result)
            return OcrResult(text="root-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    assert captured
    return captured[0]


def test_validate_fan_out_id_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        validate_fan_out_id("   ")


def test_validate_fan_out_item_id_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="item_id must be str"):
        validate_fan_out_item_id(42)


def test_fan_out_request_rejects_empty_fan_out_id() -> None:
    task_scope = mint_task_id()
    service = BoundedMultiAgentFanOutService(
        coordination=_build_coordination_service(
            build_delegated_harness(
                candidates=(
                    _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
                ),
            ),
        ),
    )
    with pytest.raises(InvalidFanOutError):
        asyncio.run(
            service.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("   "),
                    items=(
                        _fan_out_item(
                            item_id="item-a",
                            task_scope=task_scope,
                            coordination_id="coord-a",
                            delegation_id="delegation-a",
                            lease_id="lease-a",
                        ),
                    ),
                    max_concurrency=1,
                ),
                principal=admin_test_principal(),
            ),
        )


def test_fan_out_request_rejects_empty_items() -> None:
    service = BoundedMultiAgentFanOutService(
        coordination=_build_coordination_service(
            build_delegated_harness(
                candidates=(
                    _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
                ),
            ),
        ),
    )
    with pytest.raises(InvalidFanOutError, match="non-empty"):
        asyncio.run(
            service.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("fan-out-empty"),
                    items=(),
                    max_concurrency=1,
                ),
                principal=admin_test_principal(),
            ),
        )


@pytest.mark.parametrize("max_concurrency", [0, -1])
def test_fan_out_request_rejects_non_positive_concurrency(max_concurrency: int) -> None:
    task_scope = mint_task_id()
    service = BoundedMultiAgentFanOutService(
        coordination=_build_coordination_service(
            build_delegated_harness(
                candidates=(
                    _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
                ),
            ),
        ),
    )
    with pytest.raises(InvalidFanOutError, match="positive"):
        asyncio.run(
            service.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("fan-out-zero"),
                    items=(
                        _fan_out_item(
                            item_id="item-a",
                            task_scope=task_scope,
                            coordination_id="coord-a",
                            delegation_id="delegation-a",
                            lease_id="lease-a",
                        ),
                    ),
                    max_concurrency=max_concurrency,
                ),
                principal=admin_test_principal(),
            ),
        )


def test_fan_out_request_rejects_duplicate_item_ids() -> None:
    task_scope = mint_task_id()
    service = BoundedMultiAgentFanOutService(
        coordination=_build_coordination_service(
            build_delegated_harness(
                candidates=(
                    _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
                ),
            ),
        ),
    )
    with pytest.raises(InvalidFanOutError, match="duplicate"):
        asyncio.run(
            service.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("fan-out-dup"),
                    items=(
                        _fan_out_item(
                            item_id="item-a",
                            task_scope=task_scope,
                            coordination_id="coord-a",
                            delegation_id="delegation-a",
                            lease_id="lease-a",
                        ),
                        _fan_out_item(
                            item_id="item-a",
                            task_scope=task_scope,
                            coordination_id="coord-b",
                            delegation_id="delegation-b",
                            lease_id="lease-b",
                        ),
                    ),
                    max_concurrency=2,
                ),
                principal=admin_test_principal(),
            ),
        )


def test_fan_out_request_rejects_invalid_item_id() -> None:
    task_scope = mint_task_id()
    service = BoundedMultiAgentFanOutService(
        coordination=_build_coordination_service(
            build_delegated_harness(
                candidates=(
                    _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
                ),
            ),
        ),
    )
    with pytest.raises(InvalidFanOutError):
        asyncio.run(
            service.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("fan-out-invalid-item"),
                    items=(
                        FanOutItem(
                            item_id=FanOutItemId("   "),
                            request=_coordination_request(
                                task_scope=task_scope,
                                coordination_id="coord-a",
                                delegation_id="delegation-a",
                                lease_id="lease-a",
                            ),
                            delegation=CoordinationDelegation(
                                payload=OcrRequest(document_ref="doc-a"),
                            ),
                        ),
                    ),
                    max_concurrency=1,
                ),
                principal=admin_test_principal(),
            ),
        )


def test_fan_out_request_rejects_concurrency_above_platform_limit() -> None:
    task_scope = mint_task_id()
    service = BoundedMultiAgentFanOutService(
        coordination=_build_coordination_service(
            build_delegated_harness(
                candidates=(
                    _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
                ),
            ),
        ),
    )
    with pytest.raises(InvalidFanOutError, match=str(MAX_FAN_OUT_CONCURRENCY)):
        asyncio.run(
            service.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("fan-out-limit"),
                    items=(
                        _fan_out_item(
                            item_id="item-a",
                            task_scope=task_scope,
                            coordination_id="coord-a",
                            delegation_id="delegation-a",
                            lease_id="lease-a",
                        ),
                    ),
                    max_concurrency=MAX_FAN_OUT_CONCURRENCY + 1,
                ),
                principal=admin_test_principal(),
            ),
        )


@dataclass
class _ConcurrencyTracker:
    active: int = 0
    peak: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def enter(self) -> None:
        async with self.lock:
            self.active += 1
            if self.active > self.peak:
                self.peak = self.active

    async def leave(self) -> None:
        async with self.lock:
            self.active -= 1


class _DelayedOcrDelegate:
    def __init__(self, *, tracker: _ConcurrencyTracker, delay_s: float) -> None:
        self._tracker = tracker
        self._delay_s = delay_s

    async def execute(self, request: OcrRequest) -> OcrResult:
        await self._tracker.enter()
        try:
            await asyncio.sleep(self._delay_s)
            return OcrResult(text=f"ocr:{request.document_ref}")
        finally:
            await self._tracker.leave()


@pytest.mark.asyncio
async def test_fan_out_enforces_bounded_concurrency() -> None:
    tracker = _ConcurrencyTracker()
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_DelayedOcrDelegate(tracker=tracker, delay_s=0.05),
    )
    task_scope = mint_task_id()
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
    await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=items,
        max_concurrency=3,
    )
    assert tracker.peak <= 3
    assert tracker.peak > 1


@dataclass
class _TrackingCoordinationService:
    inner: MultiAgentCoordinationService[OcrRequest, OcrResult]
    calls: int = 0

    async def coordinate(self, request, *, delegation, principal):
        self.calls += 1
        return await self.inner.coordinate(
            request,
            delegation=delegation,
            principal=principal,
        )


@pytest.mark.asyncio
async def test_fan_out_uses_coordination_service_not_direct_runner() -> None:
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    inner = _build_coordination_service(harness)
    tracker = _TrackingCoordinationService(inner=inner)
    fan_out = BoundedMultiAgentFanOutService(coordination=tracker)
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()
    items = (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="item-b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
    )

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            await fan_out.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("fan-out-track"),
                    items=items,
                    max_concurrency=2,
                ),
                principal=admin_test_principal(),
            )
            return OcrResult(text="done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    assert tracker.calls == 2


class _OrderedCompletionDelegate:
    def __init__(self, *, delays: dict[str, float]) -> None:
        self._delays = delays
        self.completion_order: list[str] = []

    async def execute(self, request: OcrRequest) -> OcrResult:
        await asyncio.sleep(self._delays[request.document_ref])
        self.completion_order.append(request.document_ref)
        return OcrResult(text=f"ocr:{request.document_ref}")


@pytest.mark.asyncio
async def test_fan_out_preserves_request_order_despite_completion_order() -> None:
    delegate = _OrderedCompletionDelegate(
        delays={
            "doc-a": 0.03,
            "doc-b": 0.01,
            "doc-c": 0.02,
        },
    )
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=delegate,
    )
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="item-b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
        _fan_out_item(
            item_id="item-c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
            document_ref="doc-c",
        ),
    )
    result = await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=items,
        max_concurrency=3,
    )
    assert delegate.completion_order == ["doc-b", "doc-c", "doc-a"]
    assert [item.item_id for item in result.items] == [
        FanOutItemId("item-a"),
        FanOutItemId("item-b"),
        FanOutItemId("item-c"),
    ]
    assert [item.result.result.text for item in result.items if item.result] == [
        "ocr:doc-a",
        "ocr:doc-b",
        "ocr:doc-c",
    ]


class _FailOnDocumentDelegate:
    async def execute(self, request: OcrRequest) -> OcrResult:
        if request.document_ref == "doc-b":
            raise RuntimeError("specialist failed")
        return OcrResult(text=f"ocr:{request.document_ref}")


@pytest.mark.asyncio
async def test_fan_out_partial_failure_preserves_other_results() -> None:
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_FailOnDocumentDelegate(),
    )
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="item-b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
        _fan_out_item(
            item_id="item-c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
            document_ref="doc-c",
        ),
    )
    result = await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=items,
        max_concurrency=3,
    )
    assert result.any_failed
    assert not result.all_succeeded
    assert result.items[0].status is FanOutItemStatus.SUCCESS
    assert result.items[1].status is FanOutItemStatus.FAILURE
    assert result.items[1].failure is not None
    assert (
        result.items[1].failure.failure_code
        is CoordinationFailureCode.CHILD_EXECUTION_FAILED
    )
    assert result.items[2].status is FanOutItemStatus.SUCCESS
    assert result.items[0].result is not None
    assert result.items[2].result is not None


@pytest.mark.asyncio
async def test_fan_out_no_eligible_specialist_is_item_failure_not_operation_error() -> None:
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("legal.analysis",)),
        ),
    )
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
    )
    result = await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=items,
        max_concurrency=1,
    )
    assert result.any_failed
    assert result.items[0].failure is not None
    assert (
        result.items[0].failure.failure_code
        is CoordinationFailureCode.NO_ELIGIBLE_SPECIALIST
    )


@pytest.mark.asyncio
async def test_fan_out_releases_leases_after_success() -> None:
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="item-b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
    )
    await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=items,
        max_concurrency=2,
    )
    for lease_id in ("lease-a", "lease-b"):
        lease = harness.lease_store.get(TaskScopedAgentLeaseId(lease_id))
        assert lease is not None
        assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED


@pytest.mark.asyncio
async def test_fan_out_releases_lease_after_child_failure() -> None:
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_FailOnDocumentDelegate(),
    )
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="item-b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
    )
    await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=items,
        max_concurrency=2,
    )
    lease = harness.lease_store.get(TaskScopedAgentLeaseId("lease-b"))
    assert lease is not None
    assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED


@pytest.mark.asyncio
async def test_fan_out_child_execution_path_preserved() -> None:
    parent_execution_id = None
    child_execution_ids: list[str] = []

    class LineageDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            child_execution_ids.append(require_active_execution_id())
            return OcrResult(text=request.document_ref)

    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=LineageDelegate(),
    )
    task_scope = mint_task_id()
    root = _root_identity()
    parent_execution_id = root.execution_id
    items = (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
    )
    await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=items,
        max_concurrency=1,
    )
    assert parent_execution_id is not None
    assert len(child_execution_ids) == 1
    assert child_execution_ids[0] != parent_execution_id


@pytest.mark.asyncio
async def test_fan_out_budget_escalation_still_blocked() -> None:
    from intergrax.runtime.execution.budget.models import ExecutionBudgetReservationError

    ledger = create_execution_budget_ledger(RunBudget(max_tool_calls=50))
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    harness.service._child_execution = as_child_execution_port(
        ChildExecutionRunner[OcrRequest, OcrResult](ledger=ledger),
    )
    task_scope = mint_task_id()
    items = (
        FanOutItem(
            item_id=FanOutItemId("item-a"),
            request=_coordination_request(
                task_scope=task_scope,
                coordination_id="coord-a",
                delegation_id="delegation-a",
                lease_id="lease-a",
            ),
            delegation=CoordinationDelegation(
                payload=OcrRequest(document_ref="doc-a"),
                requested_budget=RunBudget(max_tool_calls=70),
            ),
        ),
    )
    fan_out = _build_fan_out_service(harness)
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=ledger,
            )
            try:
                result = await fan_out.fan_out(
                    FanOutRequest(
                        fan_out_id=FanOutId("fan-out-budget"),
                        items=items,
                        max_concurrency=1,
                    ),
                    principal=admin_test_principal(),
                )
            finally:
                reset_active_execution_budget(budget_token)
            assert result.items[0].failure is not None
            assert (
                result.items[0].failure.failure_code
                is CoordinationFailureCode.CHILD_EXECUTION_FAILED
            )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="budget"))


async def _run_fan_out_with_authority(
    harness,
    *,
    task_scope,
    items,
    authority: ParentExecutionAuthority,
):
    fan_out = _build_fan_out_service(harness)
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()
    captured = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            result = await fan_out.fan_out(
                FanOutRequest(
                    fan_out_id=FanOutId("fan-out-auth"),
                    items=items,
                    max_concurrency=1,
                ),
                principal=admin_test_principal(),
            )
            captured.append(result)
            return OcrResult(text="done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=authority,
    ).execute(OcrRequest(document_ref="root"))
    assert captured
    return captured[0]


@pytest.mark.asyncio
async def test_fan_out_permission_escalation_still_blocked() -> None:
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    task_scope = mint_task_id()
    items = (
        FanOutItem(
            item_id=FanOutItemId("item-a"),
            request=_coordination_request(
                task_scope=task_scope,
                coordination_id="coord-a",
                delegation_id="delegation-a",
                lease_id="lease-a",
            ),
            delegation=CoordinationDelegation(
                payload=OcrRequest(document_ref="doc-a"),
                requested_permission_scopes=("read", "delete"),
            ),
        ),
    )
    result = await _run_fan_out_with_authority(
        harness,
        task_scope=task_scope,
        items=items,
        authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    assert result.items[0].status is FanOutItemStatus.FAILURE
    assert result.items[0].failure is not None
    assert (
        result.items[0].failure.failure_code
        is CoordinationFailureCode.CHILD_EXECUTION_FAILED
    )


@pytest.mark.asyncio
async def test_npsc5a_single_coordination_regression_unchanged() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    task_scope = mint_task_id()
    coordination = _build_coordination_service(harness)
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await coordination.coordinate(
                _coordination_request(task_scope=task_scope),
                delegation=CoordinationDelegation(payload=request),
                principal=admin_test_principal(),
            )
            return result.result

    result = await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="doc-1"))
    assert result.text == "ocr:doc-1"
