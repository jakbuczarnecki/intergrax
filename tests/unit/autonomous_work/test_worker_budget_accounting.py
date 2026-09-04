# © Artur Czarnecki. All rights reserved.

"""AW-5B — worker accounting windows tests."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerDefinitionRepository,
    InMemoryWorkerInstanceRepository,
)
from intergrax.autonomous_work.in_memory_worker_accounting_repository import (
    InMemoryWorkerAccountingRepository,
)
from intergrax.autonomous_work.worker_accounting_windows import (
    daily_window_bounds,
    monthly_window_bounds,
    worker_accounting_window,
)
from intergrax.autonomous_work.worker_budget_admission import WorkerBudgetAdmissionService
from intergrax.autonomous_work.worker_budget_ports import WorkerBudgetProfileResolutionError
from intergrax.autonomous_work.worker_budget_profile_resolver import (
    MappingWorkerBudgetProfileResolver,
)
from intergrax.autonomous_work.worker_execution_accounting import (
    WorkerAccountingConflict,
    WorkerAccountingNotFound,
    WorkerExecutionAccountingService,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionDispatchDisposition,
    WorkerExecutionDispatchRejectionReason,
    WorkerExecutionDispatchRequest,
    WorkerExecutionSource,
    WorkerExecutionSourceKind,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    ProfileVersion,
)
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    WorkerAccountingWindowKind,
    WorkerBudgetAdmissionDisposition,
    WorkerBudgetAdmissionReason,
    WorkerBudgetPolicy,
    WorkerBudgetReserveRequest,
    WorkerExecutionReservationState,
    WorkerLogicalDispatchRef,
)
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.runtime.execution.budget.models import BudgetUsageTotals
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from tests.unit.autonomous_work import repository_contracts as contract_suite
from intergrax.runtime.governance.root_execution_authority_admission import (
    DenyingRootExecutionAuthorityAdmission,
)
from intergrax.contracts.autonomous_work import WorkerLifecycleState, initial_revision
from tests.unit.autonomous_work.test_worker_execution_dispatch import (
    _dispatch_request,
    _dispatch_service,
    _seed_binding_and_authority,
)
pytestmark = pytest.mark.unit

_UTC = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
_PROFILE_V1 = BudgetProfileRef(profile_id="budget/test", version=ProfileVersion(1))
_PROFILE_V2 = BudgetProfileRef(profile_id="budget/test", version=ProfileVersion(2))
_POLICY_V1 = WorkerBudgetPolicy(
    daily_execution_limit=10,
    monthly_execution_limit=100,
    max_concurrent_executions=5,
)
_POLICY_V2 = WorkerBudgetPolicy(
    daily_execution_limit=2,
    monthly_execution_limit=100,
    max_concurrent_executions=5,
)


def _logical_dispatch(
    *,
    worker_id: str,
    source_ref: str = "dispatch-1",
    source_kind: WorkerExecutionSourceKind = WorkerExecutionSourceKind.OPERATOR,
) -> WorkerLogicalDispatchRef:
    return WorkerLogicalDispatchRef(
        worker_instance_id=worker_id,
        source_kind=source_kind,
        source_ref=source_ref,
    )


def _reserve_request(
    *,
    worker_id: str,
    source_ref: str,
    policy: WorkerBudgetPolicy,
    profile_ref: BudgetProfileRef = _PROFILE_V1,
    reserved_at: datetime = _UTC,
    source_kind: WorkerExecutionSourceKind = WorkerExecutionSourceKind.OPERATOR,
) -> WorkerBudgetReserveRequest:
    return WorkerBudgetReserveRequest(
        logical_dispatch=_logical_dispatch(
            worker_id=worker_id,
            source_ref=source_ref,
            source_kind=source_kind,
        ),
        budget_profile_ref=profile_ref,
        policy=policy,
        source_kind=source_kind,
        reserved_at=reserved_at,
    )


def _budget_stack(
    *,
    policy: WorkerBudgetPolicy = _POLICY_V1,
    profile_versions: dict[tuple[str, int], WorkerBudgetPolicy] | None = None,
) -> tuple[
    InMemoryWorkerAccountingRepository,
    WorkerBudgetAdmissionService,
    InMemoryWorkerDefinitionRepository,
    InMemoryWorkerInstanceRepository,
]:
    accounting = InMemoryWorkerAccountingRepository()
    definition_repo = InMemoryWorkerDefinitionRepository()
    instance_repo = InMemoryWorkerInstanceRepository()
    versions = (
        profile_versions
        if profile_versions is not None
        else {("budget/test", 1): policy}
    )
    admission = WorkerBudgetAdmissionService(
        worker_definition_repository=definition_repo,
        profile_resolver=MappingWorkerBudgetProfileResolver(versions),
        accounting_repository=accounting,
    )
    accounting_service = WorkerExecutionAccountingService(
        accounting_repository=accounting,
    )
    return accounting, admission, definition_repo, instance_repo


def _seed_worker(
    *,
    definition_repo: InMemoryWorkerDefinitionRepository,
    instance_repo: InMemoryWorkerInstanceRepository,
    worker_id: str,
    budget_profile_ref: BudgetProfileRef = _PROFILE_V1,
) -> None:
    definition = contract_suite.worker_definition(budget_profile_ref=budget_profile_ref)
    definition_repo.create(definition)
    instance_repo.create(
        contract_suite.worker_instance(
            worker_instance_id=worker_id,
            worker_definition_id=definition.worker_definition_id,
            definition_revision=definition.revision,
            lifecycle_state=WorkerLifecycleState.ACTIVE,
            revision=initial_revision(),
        )
    )


def test_daily_limit_enforced() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(daily_execution_limit=2, max_concurrent_executions=10)
    for index in range(2):
        result = accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref=f"d-{index}", policy=policy)
        )
        assert result.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    denied = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="d-3", policy=policy)
    )
    assert denied.disposition is WorkerBudgetAdmissionDisposition.DENIED
    assert denied.evidence.reason is WorkerBudgetAdmissionReason.DAILY_LIMIT_EXCEEDED


def test_monthly_limit_enforced() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(monthly_execution_limit=2, max_concurrent_executions=10)
    for index in range(2):
        result = accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref=f"m-{index}", policy=policy)
        )
        assert result.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    denied = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="m-3", policy=policy)
    )
    assert denied.disposition is WorkerBudgetAdmissionDisposition.DENIED
    assert denied.evidence.reason is WorkerBudgetAdmissionReason.MONTHLY_LIMIT_EXCEEDED


def test_concurrency_limit_active_and_release() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=1)
    first = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="a", policy=policy)
    )
    assert first.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    denied = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="b", policy=policy)
    )
    assert denied.disposition is WorkerBudgetAdmissionDisposition.DENIED
    assert denied.evidence.reason is WorkerBudgetAdmissionReason.CONCURRENCY_LIMIT_EXCEEDED
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="a"),
        execution_id=execution_id,
        bound_at=_UTC,
    )
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        released_at=_UTC + timedelta(seconds=1),
    )
    allowed = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="b", policy=policy)
    )
    assert allowed.disposition is WorkerBudgetAdmissionDisposition.ALLOWED


def test_same_logical_dispatch_is_idempotent() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(daily_execution_limit=1, max_concurrent_executions=1)
    first = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="same", policy=policy)
    )
    second = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="same", policy=policy)
    )
    assert first.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    assert second.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    assert first.reservation == second.reservation


def test_conflicting_logical_dispatch_profile() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(daily_execution_limit=5)
    accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="same",
            policy=policy,
            profile_ref=_PROFILE_V1,
        )
    )
    conflict = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="same",
            policy=policy,
            profile_ref=_PROFILE_V2,
        )
    )
    assert conflict.disposition is WorkerBudgetAdmissionDisposition.CONFLICT


def test_profile_missing_fails_closed() -> None:
    accounting, admission, definition_repo, instance_repo = _budget_stack(
        profile_versions={},
        policy=_POLICY_V1,
    )
    worker_id = contract_suite.mint_worker_instance_id()
    _seed_worker(
        definition_repo=definition_repo,
        instance_repo=instance_repo,
        worker_id=worker_id,
    )
    worker = instance_repo.get(worker_instance_id=worker_id)
    assert worker is not None
    with pytest.raises(WorkerBudgetProfileResolutionError):
        admission.admit_dispatch(
            worker=worker,
            request=_dispatch_request(worker_id=worker_id),
        )


def test_profile_version_preserved_on_existing_reservation() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    first = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="v1",
            policy=_POLICY_V1,
            profile_ref=_PROFILE_V1,
        )
    )
    assert first.reservation is not None
    assert first.reservation.budget_profile_ref == _PROFILE_V1
    second = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="v2",
            policy=_POLICY_V2,
            profile_ref=_PROFILE_V2,
        )
    )
    assert second.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    assert second.reservation is not None
    assert second.reservation.budget_profile_ref == _PROFILE_V2
    replay = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="v1",
            policy=_POLICY_V1,
            profile_ref=_PROFILE_V1,
        )
    )
    assert replay.reservation == first.reservation
    assert replay.reservation.budget_profile_ref == _PROFILE_V1


def test_day_and_month_rollover_boundaries() -> None:
    late_day = datetime(2026, 9, 4, 23, 59, tzinfo=UTC)
    next_day = datetime(2026, 9, 5, 0, 1, tzinfo=UTC)
    daily_start, daily_end = daily_window_bounds(late_day)
    assert daily_start == datetime(2026, 9, 4, 0, 0, tzinfo=UTC)
    assert daily_end == datetime(2026, 9, 5, 0, 0, tzinfo=UTC)
    next_daily_start, _ = daily_window_bounds(next_day)
    assert next_daily_start == datetime(2026, 9, 5, 0, 0, tzinfo=UTC)
    month_start, month_end = monthly_window_bounds(late_day)
    assert month_start == datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    assert month_end == datetime(2026, 10, 1, 0, 0, tzinfo=UTC)


def test_day_rollover_resets_daily_counter() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(daily_execution_limit=1, max_concurrent_executions=5)
    late = datetime(2026, 9, 4, 23, 59, tzinfo=UTC)
    first = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="late",
            policy=policy,
            reserved_at=late,
        )
    )
    assert first.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="late"),
        execution_id=execution_id,
        bound_at=late,
    )
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        released_at=late + timedelta(minutes=1),
    )
    next_day = datetime(2026, 9, 5, 0, 1, tzinfo=UTC)
    allowed = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="early",
            policy=policy,
            reserved_at=next_day,
        )
    )
    assert allowed.disposition is WorkerBudgetAdmissionDisposition.ALLOWED


def test_execution_failure_still_counts_and_releases_concurrency() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=1)
    reserve = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="fail", policy=policy)
    )
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="fail"),
        execution_id=execution_id,
        bound_at=_UTC,
    )
    daily = worker_accounting_window(
        worker_instance_id=worker_id,
        window_kind=WorkerAccountingWindowKind.DAILY,
        at=_UTC,
    )
    state = accounting.get_window_state(window=daily)
    assert state is not None
    assert state.execution_count == 1
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        released_at=_UTC,
    )
    allowed = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="next", policy=policy)
    )
    assert allowed.disposition is WorkerBudgetAdmissionDisposition.ALLOWED


def test_duplicate_terminal_release_is_idempotent() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=1)
    accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="dup", policy=policy)
    )
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="dup"),
        execution_id=execution_id,
        bound_at=_UTC,
    )
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        released_at=_UTC,
    )
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        released_at=_UTC,
    )


def test_wrong_execution_release_not_found() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    with pytest.raises(WorkerAccountingNotFound):
        accounting.release_execution(
            worker_instance_id=contract_suite.mint_worker_instance_id(),
            execution_id=mint_execution_id(),
            released_at=_UTC,
        )


def test_cross_worker_release_rejected() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_a = contract_suite.mint_worker_instance_id()
    worker_b = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=2)
    accounting.reserve(
        _reserve_request(worker_id=worker_a, source_ref="x", policy=policy)
    )
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_a, source_ref="x"),
        execution_id=execution_id,
        bound_at=_UTC,
    )
    with pytest.raises(WorkerAccountingConflict):
        accounting.release_execution(
            worker_instance_id=worker_b,
            execution_id=execution_id,
            released_at=_UTC,
        )


def test_duplicate_usage_idempotent_conflicting_payload_rejected() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=2)
    accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="usage", policy=policy)
    )
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="usage"),
        execution_id=execution_id,
        bound_at=_UTC,
    )
    usage_a = BudgetUsageTotals(total_tokens=10)
    usage_b = BudgetUsageTotals(total_tokens=20)
    accounting.record_consumption(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        usage=usage_a,
        recorded_at=_UTC,
    )
    accounting.record_consumption(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        usage=usage_a,
        recorded_at=_UTC,
    )
    with pytest.raises(WorkerAccountingConflict):
        accounting.record_consumption(
            worker_instance_id=worker_id,
            execution_id=execution_id,
            usage=usage_b,
            recorded_at=_UTC,
        )


def test_restart_preserves_accounting_snapshot() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(daily_execution_limit=1, max_concurrent_executions=2)
    accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="persist", policy=policy)
    )
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="persist"),
        execution_id=execution_id,
        bound_at=_UTC,
    )
    restored = InMemoryWorkerAccountingRepository.from_snapshot(accounting.to_snapshot())
    denied = restored.reserve(
        _reserve_request(worker_id=worker_id, source_ref="after-restart", policy=policy)
    )
    assert denied.disposition is WorkerBudgetAdmissionDisposition.DENIED
    assert denied.evidence.reason is WorkerBudgetAdmissionReason.DAILY_LIMIT_EXCEEDED
    assert denied.evidence.reason is WorkerBudgetAdmissionReason.DAILY_LIMIT_EXCEEDED


def test_concurrent_reserve_preserves_both_or_rejects() -> None:
    accounting = InMemoryWorkerAccountingRepository()
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=1)
    outcomes: list[WorkerBudgetAdmissionDisposition] = []
    lock = threading.Barrier(2)

    def _attempt(source_ref: str) -> None:
        lock.wait(timeout=5)
        result = accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref=source_ref, policy=policy)
        )
        outcomes.append(result.disposition)

    threads = [
        threading.Thread(target=_attempt, args=("t1",)),
        threading.Thread(target=_attempt, args=("t2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    allowed = [
        item for item in outcomes if item is WorkerBudgetAdmissionDisposition.ALLOWED
    ]
    denied = [
        item for item in outcomes if item is WorkerBudgetAdmissionDisposition.DENIED
    ]
    assert len(allowed) == 1
    assert len(denied) == 1


@pytest.mark.asyncio
async def test_authority_denial_releases_budget_reservation() -> None:
    worker_id, binding_repo, membership_repo, authority_repo, delegation_repo = (
        _seed_binding_and_authority()
    )
    accounting = InMemoryWorkerAccountingRepository()
    definition_repo = InMemoryWorkerDefinitionRepository()
    instance_repo = InMemoryWorkerInstanceRepository()
    definition = contract_suite.worker_definition()
    definition_repo.create(definition)
    instance_repo.create(
        contract_suite.worker_instance(
            worker_instance_id=worker_id,
            worker_definition_id=definition.worker_definition_id,
            definition_revision=definition.revision,
            lifecycle_state=WorkerLifecycleState.ACTIVE,
            revision=initial_revision(),
        )
    )
    admission = WorkerBudgetAdmissionService(
        worker_definition_repository=definition_repo,
        profile_resolver=MappingWorkerBudgetProfileResolver(
            {("budget/default", 0): WorkerBudgetPolicy(max_concurrent_executions=1)}
        ),
        accounting_repository=accounting,
    )
    accounting_service = WorkerExecutionAccountingService(accounting_repository=accounting)
    service, intake = _dispatch_service(
        worker_repo=instance_repo,
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        root_admission=DenyingRootExecutionAuthorityAdmission(),
    )
    service._budget_admission_service = admission
    service._execution_accounting_service = accounting_service

    first = await service.dispatch(_dispatch_request(worker_id=worker_id, source_ref="one"))
    assert first.disposition is WorkerExecutionDispatchDisposition.REJECTED
    assert first.rejection_reason is WorkerExecutionDispatchRejectionReason.RUNTIME_AUTHORITY_DENIED
    second = await service.dispatch(_dispatch_request(worker_id=worker_id, source_ref="two"))
    assert second.disposition is WorkerExecutionDispatchDisposition.REJECTED
    assert len(intake.calls) == 0
