# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker budget admission service (AW-5B)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from intergrax.autonomous_work.repository import WorkerDefinitionRepository
from intergrax.autonomous_work.worker_budget_ports import (
    WorkerAccountingRepository,
    WorkerBudgetProfileResolutionError,
    WorkerBudgetProfileResolver,
)
from intergrax.autonomous_work.worker_budget_profile_resolver import (
    StaticWorkerBudgetProfileResolver,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionDispatchRequest,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.worker import WorkerInstance
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    BudgetUsageTotals,
    WorkerAccountingState,
    WorkerAccountingWindowKind,
    WorkerBudgetAdmissionDisposition,
    WorkerBudgetAdmissionEvidence,
    WorkerBudgetAdmissionReason,
    WorkerBudgetAdmissionResult,
    WorkerBudgetPolicy,
    WorkerBudgetReserveRequest,
    WorkerLogicalDispatchRef,
    WorkerProactiveEvaluationAccountingRequest,
)

from intergrax.autonomous_work.worker_accounting_windows import worker_accounting_window


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WorkerBudgetAdmissionService:
    """Resolve worker budget policy and atomically reserve accounting quotas."""

    def __init__(
        self,
        *,
        worker_definition_repository: WorkerDefinitionRepository,
        profile_resolver: WorkerBudgetProfileResolver,
        accounting_repository: WorkerAccountingRepository,
        default_policy: WorkerBudgetPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._worker_definition_repository = worker_definition_repository
        self._profile_resolver = profile_resolver
        self._accounting_repository = accounting_repository
        self._default_policy = default_policy
        self._clock = clock or _utc_now
        if default_policy is not None and not isinstance(
            profile_resolver,
            StaticWorkerBudgetProfileResolver,
        ):
            self._fallback_resolver = StaticWorkerBudgetProfileResolver(default_policy)
        else:
            self._fallback_resolver = None

    def admit_dispatch(
        self,
        *,
        worker: WorkerInstance,
        request: WorkerExecutionDispatchRequest[object, object],
    ) -> WorkerBudgetAdmissionResult:
        evaluated_at = request.requested_at
        profile_ref, policy = self._resolve_profile_for_worker(worker)
        logical_dispatch = WorkerLogicalDispatchRef(
            worker_instance_id=request.worker_instance_id,
            source_kind=request.source.source_kind,
            source_ref=request.source.source_ref,
        )
        try:
            return self._accounting_repository.reserve(
                WorkerBudgetReserveRequest(
                    logical_dispatch=logical_dispatch,
                    budget_profile_ref=profile_ref,
                    policy=policy,
                    source_kind=request.source.source_kind,
                    reserved_at=evaluated_at,
                )
            )
        except Exception:
            return self._unavailable_result(
                worker=worker,
                profile_ref=profile_ref,
                policy=policy,
                evaluated_at=evaluated_at,
            )

    def record_proactive_evaluation(
        self,
        *,
        worker: WorkerInstance,
        evaluation_ref: str,
        evaluated_at: datetime | None = None,
    ) -> WorkerBudgetAdmissionResult:
        resolved_at = evaluated_at or self._clock()
        profile_ref, policy = self._resolve_profile_for_worker(worker)
        try:
            return self._accounting_repository.record_proactive_evaluation(
                WorkerProactiveEvaluationAccountingRequest(
                    worker_instance_id=worker.worker_instance_id,
                    budget_profile_ref=profile_ref,
                    policy=policy,
                    evaluation_ref=evaluation_ref,
                    evaluated_at=resolved_at,
                )
            )
        except Exception:
            return self._unavailable_result(
                worker=worker,
                profile_ref=profile_ref,
                policy=policy,
                evaluated_at=resolved_at,
            )

    def _resolve_profile_for_worker(
        self,
        worker: WorkerInstance,
    ) -> tuple[BudgetProfileRef, WorkerBudgetPolicy]:
        definition = self._worker_definition_repository.get(
            worker_definition_id=worker.worker_definition_id,
            definition_revision=worker.definition_revision,
        )
        if definition is None:
            if self._fallback_resolver is None:
                raise WorkerBudgetProfileResolutionError(
                    "worker definition not found for budget profile resolution"
                )
            profile_ref = BudgetProfileRef(
                profile_id="platform/default",
                version=initial_profile_version(),
            )
            return profile_ref, self._fallback_resolver.resolve(profile_ref)
        profile_ref = definition.budget_profile_ref
        try:
            return profile_ref, self._profile_resolver.resolve(profile_ref)
        except WorkerBudgetProfileResolutionError:
            if self._fallback_resolver is not None:
                return profile_ref, self._fallback_resolver.resolve(profile_ref)
            raise

    def _unavailable_result(
        self,
        *,
        worker: WorkerInstance,
        profile_ref: BudgetProfileRef,
        policy: WorkerBudgetPolicy,
        evaluated_at: datetime,
    ) -> WorkerBudgetAdmissionResult:
        daily_window = worker_accounting_window(
            worker_instance_id=worker.worker_instance_id,
            window_kind=WorkerAccountingWindowKind.DAILY,
            at=evaluated_at,
        )
        monthly_window = worker_accounting_window(
            worker_instance_id=worker.worker_instance_id,
            window_kind=WorkerAccountingWindowKind.MONTHLY,
            at=evaluated_at,
        )
        empty_daily = self._accounting_repository.get_window_state(window=daily_window)
        empty_monthly = self._accounting_repository.get_window_state(window=monthly_window)
        daily_state = empty_daily or WorkerAccountingState(
            window=daily_window,
            revision=0,
            execution_count=0,
            reserved_dispatch_count=0,
            recovery_execution_count=0,
            codecraft_execution_count=0,
            proactive_evaluation_count=0,
            aggregate_usage=BudgetUsageTotals(),
        )
        monthly_state = empty_monthly or WorkerAccountingState(
            window=monthly_window,
            revision=0,
            execution_count=0,
            reserved_dispatch_count=0,
            recovery_execution_count=0,
            codecraft_execution_count=0,
            proactive_evaluation_count=0,
            aggregate_usage=BudgetUsageTotals(),
        )

        return WorkerBudgetAdmissionResult(
            disposition=WorkerBudgetAdmissionDisposition.UNAVAILABLE,
            evidence=WorkerBudgetAdmissionEvidence(
                worker_instance_id=worker.worker_instance_id,
                budget_profile_ref=profile_ref,
                daily_window=daily_window,
                monthly_window=monthly_window,
                applied_policy=policy,
                daily_state=daily_state,
                monthly_state=monthly_state,
                active_reservation_count=0,
                evaluated_at=evaluated_at,
                reason=WorkerBudgetAdmissionReason.ACCOUNTING_UNAVAILABLE,
            ),
            reservation=None,
        )
