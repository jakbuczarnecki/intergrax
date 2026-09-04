# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory reference worker accounting repository (AW-5B).

Process-local only — not production durable. Used for unit tests and contract
suite reference behavior.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import datetime

from intergrax.autonomous_work.repository import AutonomousWorkRepositoryCapabilities
from intergrax.autonomous_work.worker_accounting_windows import (
    worker_accounting_window,
    window_identity_key,
)
from intergrax.autonomous_work.worker_budget_ports import (
    WorkerAccountingConflict,
    WorkerAccountingNotFound,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionSourceKind,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    BudgetUsageTotals,
    WorkerAccountingState,
    WorkerAccountingWindow,
    WorkerAccountingWindowKind,
    WorkerBudgetAdmissionDisposition,
    WorkerBudgetAdmissionEvidence,
    WorkerBudgetAdmissionReason,
    WorkerBudgetAdmissionResult,
    WorkerBudgetReserveRequest,
    WorkerExecutionReservation,
    WorkerExecutionReservationState,
    WorkerLogicalDispatchRef,
    WorkerProactiveEvaluationAccountingRequest,
)
from intergrax.contracts.execution_identity import ExecutionId

_BACKEND_ID = "autonomous_work.worker_accounting.in_memory"
_CAPABILITIES = AutonomousWorkRepositoryCapabilities(
    backend_id=_BACKEND_ID,
    durable=False,
    reference_only=True,
)

_ReservationKey = tuple[str, str, str]


def _logical_dispatch_key(logical_dispatch: WorkerLogicalDispatchRef) -> _ReservationKey:
    return (
        logical_dispatch.worker_instance_id.strip(),
        logical_dispatch.source_kind.value,
        logical_dispatch.source_ref,
    )


def _empty_state(*, window: WorkerAccountingWindow) -> WorkerAccountingState:
    return WorkerAccountingState(
        window=window,
        revision=0,
        execution_count=0,
        reserved_dispatch_count=0,
        recovery_execution_count=0,
        codecraft_execution_count=0,
        proactive_evaluation_count=0,
        aggregate_usage=BudgetUsageTotals(),
    )


def _usage_to_json(usage: BudgetUsageTotals) -> str:
    return json.dumps(
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "llm_calls": usage.llm_calls,
            "tool_calls": usage.tool_calls,
            "rag_invocations": usage.rag_invocations,
            "websearch_invocations": usage.websearch_invocations,
            "wall_time_seconds": usage.wall_time_seconds,
            "planner_iterations": usage.planner_iterations,
            "replans": usage.replans,
        }
    )


def _usage_from_json(payload: str) -> BudgetUsageTotals:
    data = json.loads(payload)
    return BudgetUsageTotals(
        input_tokens=int(data["input_tokens"]),
        output_tokens=int(data["output_tokens"]),
        total_tokens=int(data["total_tokens"]),
        llm_calls=int(data["llm_calls"]),
        tool_calls=int(data["tool_calls"]),
        rag_invocations=int(data["rag_invocations"]),
        websearch_invocations=int(data["websearch_invocations"]),
        wall_time_seconds=float(data["wall_time_seconds"]),
        planner_iterations=int(data["planner_iterations"]),
        replans=int(data["replans"]),
    )


class InMemoryWorkerAccountingRepository:
    """Thread-safe in-memory worker accounting reference adapter."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._windows: dict[tuple[str, str, str], WorkerAccountingState] = {}
        self._reservations: dict[_ReservationKey, WorkerExecutionReservation] = {}
        self._execution_bindings: dict[str, _ReservationKey] = {}
        self._recorded_usage: dict[str, str] = {}
        self._proactive_evaluations: dict[tuple[str, str], str] = {}

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def get_window_state(
        self,
        *,
        window: WorkerAccountingWindow,
    ) -> WorkerAccountingState | None:
        with self._lock:
            return self._windows.get(window_identity_key(window))

    def reserve(self, request: WorkerBudgetReserveRequest) -> WorkerBudgetAdmissionResult:
        with self._lock:
            return self._reserve_locked(request)

    def bind_execution(
        self,
        *,
        logical_dispatch: WorkerLogicalDispatchRef,
        execution_id: ExecutionId,
        bound_at: datetime,
    ) -> WorkerExecutionReservation:
        with self._lock:
            key = _logical_dispatch_key(logical_dispatch)
            reservation = self._reservations.get(key)
            if reservation is None:
                raise WorkerAccountingNotFound(
                    f"reservation not found for {logical_dispatch.source_ref}"
                )
            if reservation.state is WorkerExecutionReservationState.RELEASED:
                raise WorkerAccountingConflict("cannot bind released reservation")
            if reservation.state is WorkerExecutionReservationState.BOUND_TO_EXECUTION:
                if reservation.execution_id == execution_id:
                    return reservation
                raise WorkerAccountingConflict(
                    "reservation already bound to different execution"
                )
            updated = replace(
                reservation,
                state=WorkerExecutionReservationState.BOUND_TO_EXECUTION,
                execution_id=execution_id,
                bound_at=bound_at,
            )
            self._reservations[key] = updated
            self._execution_bindings[execution_id.strip()] = key
            self._increment_execution_counts_locked(
                reservation=updated,
                source_kind=logical_dispatch.source_kind,
            )
            return updated

    def release_reservation(
        self,
        *,
        logical_dispatch: WorkerLogicalDispatchRef,
        released_at: datetime,
    ) -> WorkerExecutionReservation:
        with self._lock:
            key = _logical_dispatch_key(logical_dispatch)
            reservation = self._reservations.get(key)
            if reservation is None:
                raise WorkerAccountingNotFound(
                    f"reservation not found for {logical_dispatch.source_ref}"
                )
            if reservation.state is WorkerExecutionReservationState.RELEASED:
                return reservation
            if reservation.state is WorkerExecutionReservationState.BOUND_TO_EXECUTION:
                raise WorkerAccountingConflict(
                    "cannot release reservation already bound to execution"
                )
            self._decrement_reserved_dispatch_locked(reservation)
            updated = replace(
                reservation,
                state=WorkerExecutionReservationState.RELEASED,
                released_at=released_at,
            )
            self._reservations[key] = updated
            return updated

    def release_execution(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        released_at: datetime,
    ) -> WorkerExecutionReservation:
        with self._lock:
            key = self._execution_bindings.get(execution_id.strip())
            if key is None:
                raise WorkerAccountingNotFound(
                    f"execution binding not found for {execution_id}"
                )
            reservation = self._reservations[key]
            if reservation.logical_dispatch.worker_instance_id.strip() != (
                worker_instance_id.strip()
            ):
                raise WorkerAccountingConflict(
                    "execution binding belongs to a different worker"
                )
            if reservation.state is WorkerExecutionReservationState.RELEASED:
                return reservation
            updated = replace(
                reservation,
                state=WorkerExecutionReservationState.RELEASED,
                released_at=released_at,
            )
            self._reservations[key] = updated
            return updated

    def record_consumption(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        usage: BudgetUsageTotals,
        recorded_at: datetime,
    ) -> None:
        del recorded_at
        with self._lock:
            key = self._execution_bindings.get(execution_id.strip())
            if key is None:
                raise WorkerAccountingNotFound(
                    f"execution binding not found for {execution_id}"
                )
            reservation = self._reservations[key]
            if reservation.logical_dispatch.worker_instance_id.strip() != (
                worker_instance_id.strip()
            ):
                raise WorkerAccountingConflict(
                    "execution binding belongs to a different worker"
                )
            usage_json = _usage_to_json(usage)
            existing = self._recorded_usage.get(execution_id.strip())
            if existing is not None:
                if existing != usage_json:
                    raise WorkerAccountingConflict(
                        "conflicting usage payload for execution"
                    )
                return
            self._recorded_usage[execution_id.strip()] = usage_json
            for window in (reservation.daily_window, reservation.monthly_window):
                window_key = window_identity_key(window)
                state = self._windows.get(window_key) or _empty_state(window=window)
                self._windows[window_key] = replace(
                    state,
                    revision=state.revision + 1,
                    aggregate_usage=state.aggregate_usage.add(usage),
                )

    def record_proactive_evaluation(
        self,
        request: WorkerProactiveEvaluationAccountingRequest,
    ) -> WorkerBudgetAdmissionResult:
        with self._lock:
            evaluation_key = (
                request.worker_instance_id.strip(),
                request.evaluation_ref,
            )
            if evaluation_key in self._proactive_evaluations:
                stored_profile = self._proactive_evaluations[evaluation_key]
                if stored_profile != request.budget_profile_ref.profile_id:
                    return self._conflict_result(request=request)
                daily_window = worker_accounting_window(
                    worker_instance_id=request.worker_instance_id,
                    window_kind=WorkerAccountingWindowKind.DAILY,
                    at=request.evaluated_at,
                )
                monthly_window = worker_accounting_window(
                    worker_instance_id=request.worker_instance_id,
                    window_kind=WorkerAccountingWindowKind.MONTHLY,
                    at=request.evaluated_at,
                )
                daily_state = self._get_or_create_window_locked(daily_window)
                monthly_state = self._get_or_create_window_locked(monthly_window)
                return WorkerBudgetAdmissionResult(
                    disposition=WorkerBudgetAdmissionDisposition.ALLOWED,
                    evidence=self._evidence(
                        request=request,
                        daily_window=daily_window,
                        monthly_window=monthly_window,
                        daily_state=daily_state,
                        monthly_state=monthly_state,
                        active_reservation_count=self._active_reservation_count_locked(
                            request.worker_instance_id.strip()
                        ),
                    ),
                    reservation=None,
                )

            daily_window = worker_accounting_window(
                worker_instance_id=request.worker_instance_id,
                window_kind=WorkerAccountingWindowKind.DAILY,
                at=request.evaluated_at,
            )
            monthly_window = worker_accounting_window(
                worker_instance_id=request.worker_instance_id,
                window_kind=WorkerAccountingWindowKind.MONTHLY,
                at=request.evaluated_at,
            )
            daily_state = self._get_or_create_window_locked(daily_window)
            limit = request.policy.daily_proactive_evaluation_limit
            if limit is not None:
                projected = daily_state.proactive_evaluation_count + 1
                if projected > limit:
                    return self._denied_result(
                        request=request,
                        reason=WorkerBudgetAdmissionReason.PROACTIVE_LIMIT_EXCEEDED,
                        daily_window=daily_window,
                        monthly_window=monthly_window,
                        daily_state=daily_state,
                        monthly_state=self._get_or_create_window_locked(monthly_window),
                    )
            self._proactive_evaluations[evaluation_key] = (
                request.budget_profile_ref.profile_id
            )
            daily_state = self._increment_proactive_locked(daily_state)
            return WorkerBudgetAdmissionResult(
                disposition=WorkerBudgetAdmissionDisposition.ALLOWED,
                evidence=self._evidence(
                    request=request,
                    daily_window=daily_window,
                    monthly_window=monthly_window,
                    daily_state=daily_state,
                    monthly_state=self._get_or_create_window_locked(monthly_window),
                    active_reservation_count=self._active_reservation_count_locked(
                        request.worker_instance_id.strip()
                    ),
                ),
                reservation=None,
            )

    def _reserve_locked(
        self,
        request: WorkerBudgetReserveRequest,
    ) -> WorkerBudgetAdmissionResult:
        key = _logical_dispatch_key(request.logical_dispatch)
        existing = self._reservations.get(key)
        if existing is not None:
            if not self._reservation_matches_request(existing, request):
                return self._conflict_result(request=request)
            if existing.state is WorkerExecutionReservationState.RELEASED:
                return self._conflict_result(request=request)
            daily_state = self._get_or_create_window_locked(existing.daily_window)
            monthly_state = self._get_or_create_window_locked(existing.monthly_window)
            return WorkerBudgetAdmissionResult(
                disposition=WorkerBudgetAdmissionDisposition.ALLOWED,
                evidence=self._evidence(
                    request=request,
                    daily_window=existing.daily_window,
                    monthly_window=existing.monthly_window,
                    daily_state=daily_state,
                    monthly_state=monthly_state,
                    active_reservation_count=self._active_reservation_count_locked(
                        request.logical_dispatch.worker_instance_id.strip()
                    ),
                ),
                reservation=existing,
            )

        daily_window = worker_accounting_window(
            worker_instance_id=request.logical_dispatch.worker_instance_id,
            window_kind=WorkerAccountingWindowKind.DAILY,
            at=request.reserved_at,
        )
        monthly_window = worker_accounting_window(
            worker_instance_id=request.logical_dispatch.worker_instance_id,
            window_kind=WorkerAccountingWindowKind.MONTHLY,
            at=request.reserved_at,
        )
        daily_state = self._get_or_create_window_locked(daily_window)
        monthly_state = self._get_or_create_window_locked(monthly_window)
        active_count = self._active_reservation_count_locked(
            request.logical_dispatch.worker_instance_id.strip()
        )

        denial = self._check_limits_locked(
            request=request,
            daily_state=daily_state,
            monthly_state=monthly_state,
            active_count=active_count,
            daily_window=daily_window,
            monthly_window=monthly_window,
        )
        if denial is not None:
            return denial

        reservation = WorkerExecutionReservation(
            logical_dispatch=request.logical_dispatch,
            budget_profile_ref=request.budget_profile_ref,
            daily_window=daily_window,
            monthly_window=monthly_window,
            reserved_at=request.reserved_at,
            state=WorkerExecutionReservationState.RESERVED,
        )
        self._reservations[key] = reservation
        self._increment_reserved_dispatch_locked(reservation)
        daily_state = self._get_or_create_window_locked(daily_window)
        monthly_state = self._get_or_create_window_locked(monthly_window)
        return WorkerBudgetAdmissionResult(
            disposition=WorkerBudgetAdmissionDisposition.ALLOWED,
            evidence=self._evidence(
                request=request,
                daily_window=daily_window,
                monthly_window=monthly_window,
                daily_state=daily_state,
                monthly_state=monthly_state,
                active_reservation_count=self._active_reservation_count_locked(
                    request.logical_dispatch.worker_instance_id.strip()
                ),
            ),
            reservation=reservation,
        )

    def _check_limits_locked(
        self,
        *,
        request: WorkerBudgetReserveRequest,
        daily_state: WorkerAccountingState,
        monthly_state: WorkerAccountingState,
        active_count: int,
        daily_window: WorkerAccountingWindow,
        monthly_window: WorkerAccountingWindow,
    ) -> WorkerBudgetAdmissionResult | None:
        policy = request.policy
        projected_daily = (
            daily_state.execution_count + daily_state.reserved_dispatch_count + 1
        )
        if policy.daily_execution_limit is not None:
            if projected_daily > policy.daily_execution_limit:
                return self._denied_result(
                    request=request,
                    reason=WorkerBudgetAdmissionReason.DAILY_LIMIT_EXCEEDED,
                    daily_window=daily_window,
                    monthly_window=monthly_window,
                    daily_state=daily_state,
                    monthly_state=monthly_state,
                )
        projected_monthly = (
            monthly_state.execution_count + monthly_state.reserved_dispatch_count + 1
        )
        if policy.monthly_execution_limit is not None:
            if projected_monthly > policy.monthly_execution_limit:
                return self._denied_result(
                    request=request,
                    reason=WorkerBudgetAdmissionReason.MONTHLY_LIMIT_EXCEEDED,
                    daily_window=daily_window,
                    monthly_window=monthly_window,
                    daily_state=daily_state,
                    monthly_state=monthly_state,
                )
        if policy.max_concurrent_executions is not None:
            if active_count + 1 > policy.max_concurrent_executions:
                return self._denied_result(
                    request=request,
                    reason=WorkerBudgetAdmissionReason.CONCURRENCY_LIMIT_EXCEEDED,
                    daily_window=daily_window,
                    monthly_window=monthly_window,
                    daily_state=daily_state,
                    monthly_state=monthly_state,
                )
        if request.source_kind is WorkerExecutionSourceKind.RECOVERY:
            if policy.daily_recovery_execution_limit is not None:
                projected = daily_state.recovery_execution_count + 1
                if projected > policy.daily_recovery_execution_limit:
                    return self._denied_result(
                        request=request,
                        reason=WorkerBudgetAdmissionReason.RECOVERY_LIMIT_EXCEEDED,
                        daily_window=daily_window,
                        monthly_window=monthly_window,
                        daily_state=daily_state,
                        monthly_state=monthly_state,
                    )
        return None

    def _increment_execution_counts_locked(
        self,
        *,
        reservation: WorkerExecutionReservation,
        source_kind: WorkerExecutionSourceKind,
    ) -> None:
        for window in (reservation.daily_window, reservation.monthly_window):
            state = self._get_or_create_window_locked(window)
            recovery_count = state.recovery_execution_count
            if source_kind is WorkerExecutionSourceKind.RECOVERY:
                recovery_count += 1
            self._windows[window_identity_key(window)] = replace(
                state,
                revision=state.revision + 1,
                execution_count=state.execution_count + 1,
                reserved_dispatch_count=max(state.reserved_dispatch_count - 1, 0),
                recovery_execution_count=recovery_count,
            )

    def _increment_reserved_dispatch_locked(
        self,
        reservation: WorkerExecutionReservation,
    ) -> None:
        for window in (reservation.daily_window, reservation.monthly_window):
            state = self._get_or_create_window_locked(window)
            self._windows[window_identity_key(window)] = replace(
                state,
                revision=state.revision + 1,
                reserved_dispatch_count=state.reserved_dispatch_count + 1,
            )

    def _decrement_reserved_dispatch_locked(
        self,
        reservation: WorkerExecutionReservation,
    ) -> None:
        for window in (reservation.daily_window, reservation.monthly_window):
            state = self._get_or_create_window_locked(window)
            self._windows[window_identity_key(window)] = replace(
                state,
                revision=state.revision + 1,
                reserved_dispatch_count=max(state.reserved_dispatch_count - 1, 0),
            )

    def _increment_proactive_locked(
        self,
        state: WorkerAccountingState,
    ) -> WorkerAccountingState:
        updated = replace(
            state,
            revision=state.revision + 1,
            proactive_evaluation_count=state.proactive_evaluation_count + 1,
        )
        self._windows[window_identity_key(state.window)] = updated
        return updated

    def _get_or_create_window_locked(
        self,
        window: WorkerAccountingWindow,
    ) -> WorkerAccountingState:
        key = window_identity_key(window)
        state = self._windows.get(key)
        if state is None:
            state = _empty_state(window=window)
            self._windows[key] = state
        return state

    def _active_reservation_count_locked(self, worker_instance_id: str) -> int:
        count = 0
        for reservation in self._reservations.values():
            if reservation.logical_dispatch.worker_instance_id.strip() != worker_instance_id:
                continue
            if reservation.state in (
                WorkerExecutionReservationState.RESERVED,
                WorkerExecutionReservationState.BOUND_TO_EXECUTION,
            ):
                count += 1
        return count

    @staticmethod
    def _reservation_matches_request(
        reservation: WorkerExecutionReservation,
        request: WorkerBudgetReserveRequest,
    ) -> bool:
        return (
            reservation.budget_profile_ref == request.budget_profile_ref
            and reservation.logical_dispatch == request.logical_dispatch
        )

    def _evidence(
        self,
        *,
        request: WorkerBudgetReserveRequest | WorkerProactiveEvaluationAccountingRequest,
        daily_window: WorkerAccountingWindow,
        monthly_window: WorkerAccountingWindow,
        daily_state: WorkerAccountingState,
        monthly_state: WorkerAccountingState,
        active_reservation_count: int,
        reason: WorkerBudgetAdmissionReason | None = None,
    ) -> WorkerBudgetAdmissionEvidence:
        if isinstance(request, WorkerBudgetReserveRequest):
            worker_instance_id = request.logical_dispatch.worker_instance_id
            budget_profile_ref = request.budget_profile_ref
            policy = request.policy
            evaluated_at = request.reserved_at
        else:
            worker_instance_id = request.worker_instance_id
            budget_profile_ref = request.budget_profile_ref
            policy = request.policy
            evaluated_at = request.evaluated_at
        return WorkerBudgetAdmissionEvidence(
            worker_instance_id=worker_instance_id,
            budget_profile_ref=budget_profile_ref,
            daily_window=daily_window,
            monthly_window=monthly_window,
            applied_policy=policy,
            daily_state=daily_state,
            monthly_state=monthly_state,
            active_reservation_count=active_reservation_count,
            evaluated_at=evaluated_at,
            reason=reason,
        )

    def _denied_result(
        self,
        *,
        request: WorkerBudgetReserveRequest | WorkerProactiveEvaluationAccountingRequest,
        reason: WorkerBudgetAdmissionReason,
        daily_window: WorkerAccountingWindow,
        monthly_window: WorkerAccountingWindow,
        daily_state: WorkerAccountingState,
        monthly_state: WorkerAccountingState,
    ) -> WorkerBudgetAdmissionResult:
        if isinstance(request, WorkerBudgetReserveRequest):
            worker_id = request.logical_dispatch.worker_instance_id.strip()
        else:
            worker_id = request.worker_instance_id.strip()
        return WorkerBudgetAdmissionResult(
            disposition=WorkerBudgetAdmissionDisposition.DENIED,
            evidence=self._evidence(
                request=request,
                daily_window=daily_window,
                monthly_window=monthly_window,
                daily_state=daily_state,
                monthly_state=monthly_state,
                active_reservation_count=self._active_reservation_count_locked(worker_id),
                reason=reason,
            ),
            reservation=None,
        )

    def _conflict_result(
        self,
        *,
        request: WorkerBudgetReserveRequest | WorkerProactiveEvaluationAccountingRequest,
    ) -> WorkerBudgetAdmissionResult:
        if isinstance(request, WorkerBudgetReserveRequest):
            daily_window = worker_accounting_window(
                worker_instance_id=request.logical_dispatch.worker_instance_id,
                window_kind=WorkerAccountingWindowKind.DAILY,
                at=request.reserved_at,
            )
            monthly_window = worker_accounting_window(
                worker_instance_id=request.logical_dispatch.worker_instance_id,
                window_kind=WorkerAccountingWindowKind.MONTHLY,
                at=request.reserved_at,
            )
            worker_id = request.logical_dispatch.worker_instance_id.strip()
        else:
            daily_window = worker_accounting_window(
                worker_instance_id=request.worker_instance_id,
                window_kind=WorkerAccountingWindowKind.DAILY,
                at=request.evaluated_at,
            )
            monthly_window = worker_accounting_window(
                worker_instance_id=request.worker_instance_id,
                window_kind=WorkerAccountingWindowKind.MONTHLY,
                at=request.evaluated_at,
            )
            worker_id = request.worker_instance_id.strip()
        daily_state = self._get_or_create_window_locked(daily_window)
        monthly_state = self._get_or_create_window_locked(monthly_window)
        return WorkerBudgetAdmissionResult(
            disposition=WorkerBudgetAdmissionDisposition.CONFLICT,
            evidence=self._evidence(
                request=request,
                daily_window=daily_window,
                monthly_window=monthly_window,
                daily_state=daily_state,
                monthly_state=monthly_state,
                active_reservation_count=self._active_reservation_count_locked(worker_id),
            ),
            reservation=None,
        )

    def to_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "windows": {
                    "|".join(key): self._serialize_state(state)
                    for key, state in self._windows.items()
                },
                "reservations": {
                    "|".join(key): self._serialize_reservation(reservation)
                    for key, reservation in self._reservations.items()
                },
                "execution_bindings": dict(self._execution_bindings),
                "recorded_usage": dict(self._recorded_usage),
                "proactive_evaluations": {
                    "|".join(key): value
                    for key, value in self._proactive_evaluations.items()
                },
            }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> InMemoryWorkerAccountingRepository:
        repo = cls()
        windows = payload.get("windows", {})
        if isinstance(windows, dict):
            for key_text, state_payload in windows.items():
                if isinstance(state_payload, dict):
                    repo._windows[tuple(key_text.split("|", 2))] = cls._deserialize_state(
                        state_payload
                    )
        reservations = payload.get("reservations", {})
        if isinstance(reservations, dict):
            for key_text, reservation_payload in reservations.items():
                if isinstance(reservation_payload, dict):
                    repo._reservations[tuple(key_text.split("|", 2))] = (
                        cls._deserialize_reservation(reservation_payload)
                    )
        bindings = payload.get("execution_bindings", {})
        if isinstance(bindings, dict):
            repo._execution_bindings = {
                str(execution_id): tuple(str(part) for part in key)
                for execution_id, key in bindings.items()
            }
        recorded = payload.get("recorded_usage", {})
        if isinstance(recorded, dict):
            repo._recorded_usage = {str(k): str(v) for k, v in recorded.items()}
        proactive = payload.get("proactive_evaluations", {})
        if isinstance(proactive, dict):
            repo._proactive_evaluations = {
                tuple(str(part) for part in key.split("|", 1)): str(value)
                for key, value in proactive.items()
            }
        return repo

    @staticmethod
    def _serialize_state(state: WorkerAccountingState) -> dict[str, object]:
        return {
            "window": {
                "worker_instance_id": state.window.worker_instance_id.strip(),
                "window_kind": state.window.window_kind.value,
                "window_start": state.window.window_start.isoformat(),
                "window_end": state.window.window_end.isoformat(),
            },
            "revision": state.revision,
            "execution_count": state.execution_count,
            "reserved_dispatch_count": state.reserved_dispatch_count,
            "recovery_execution_count": state.recovery_execution_count,
            "codecraft_execution_count": state.codecraft_execution_count,
            "proactive_evaluation_count": state.proactive_evaluation_count,
            "aggregate_usage": _usage_to_json(state.aggregate_usage),
        }

    @staticmethod
    def _deserialize_state(payload: dict[str, object]) -> WorkerAccountingState:
        window_payload = payload["window"]
        assert isinstance(window_payload, dict)
        window = WorkerAccountingWindow(
            worker_instance_id=WorkerInstanceId(str(window_payload["worker_instance_id"])),
            window_kind=WorkerAccountingWindowKind(str(window_payload["window_kind"])),
            window_start=datetime.fromisoformat(str(window_payload["window_start"])),
            window_end=datetime.fromisoformat(str(window_payload["window_end"])),
        )
        return WorkerAccountingState(
            window=window,
            revision=int(payload["revision"]),
            execution_count=int(payload["execution_count"]),
            reserved_dispatch_count=int(payload["reserved_dispatch_count"]),
            recovery_execution_count=int(payload["recovery_execution_count"]),
            codecraft_execution_count=int(payload["codecraft_execution_count"]),
            proactive_evaluation_count=int(payload["proactive_evaluation_count"]),
            aggregate_usage=_usage_from_json(str(payload["aggregate_usage"])),
        )

    @staticmethod
    def _serialize_reservation(
        reservation: WorkerExecutionReservation,
    ) -> dict[str, object]:
        return {
            "logical_dispatch": {
                "worker_instance_id": reservation.logical_dispatch.worker_instance_id.strip(),
                "source_kind": reservation.logical_dispatch.source_kind.value,
                "source_ref": reservation.logical_dispatch.source_ref,
            },
            "budget_profile_ref": {
                "profile_id": reservation.budget_profile_ref.profile_id,
                "version": reservation.budget_profile_ref.version.value,
            },
            "daily_window": InMemoryWorkerAccountingRepository._serialize_state(
                WorkerAccountingState(
                    window=reservation.daily_window,
                    revision=0,
                    execution_count=0,
                    reserved_dispatch_count=0,
                    recovery_execution_count=0,
                    codecraft_execution_count=0,
                    proactive_evaluation_count=0,
                    aggregate_usage=BudgetUsageTotals(),
                )
            )["window"],
            "monthly_window": InMemoryWorkerAccountingRepository._serialize_state(
                WorkerAccountingState(
                    window=reservation.monthly_window,
                    revision=0,
                    execution_count=0,
                    reserved_dispatch_count=0,
                    recovery_execution_count=0,
                    codecraft_execution_count=0,
                    proactive_evaluation_count=0,
                    aggregate_usage=BudgetUsageTotals(),
                )
            )["window"],
            "reserved_at": reservation.reserved_at.isoformat(),
            "state": reservation.state.value,
            "execution_id": (
                reservation.execution_id.strip() if reservation.execution_id is not None else None
            ),
            "bound_at": (
                reservation.bound_at.isoformat() if reservation.bound_at is not None else None
            ),
            "released_at": (
                reservation.released_at.isoformat()
                if reservation.released_at is not None
                else None
            ),
        }

    @staticmethod
    def _deserialize_reservation(
        payload: dict[str, object],
    ) -> WorkerExecutionReservation:
        from intergrax.contracts.autonomous_work.profile_reference import (
            BudgetProfileRef,
            ProfileVersion,
        )

        logical_payload = payload["logical_dispatch"]
        assert isinstance(logical_payload, dict)
        profile_payload = payload["budget_profile_ref"]
        assert isinstance(profile_payload, dict)
        daily_payload = payload["daily_window"]
        monthly_payload = payload["monthly_window"]
        assert isinstance(daily_payload, dict)
        assert isinstance(monthly_payload, dict)

        def _window_from_payload(window_payload: dict[str, object]) -> WorkerAccountingWindow:
            return WorkerAccountingWindow(
                worker_instance_id=WorkerInstanceId(str(window_payload["worker_instance_id"])),
                window_kind=WorkerAccountingWindowKind(str(window_payload["window_kind"])),
                window_start=datetime.fromisoformat(str(window_payload["window_start"])),
                window_end=datetime.fromisoformat(str(window_payload["window_end"])),
            )

        execution_id_raw = payload.get("execution_id")
        bound_at_raw = payload.get("bound_at")
        released_at_raw = payload.get("released_at")
        return WorkerExecutionReservation(
            logical_dispatch=WorkerLogicalDispatchRef(
                worker_instance_id=WorkerInstanceId(str(logical_payload["worker_instance_id"])),
                source_kind=WorkerExecutionSourceKind(str(logical_payload["source_kind"])),
                source_ref=str(logical_payload["source_ref"]),
            ),
            budget_profile_ref=BudgetProfileRef(
                profile_id=str(profile_payload["profile_id"]),
                version=ProfileVersion(int(profile_payload["version"])),
            ),
            daily_window=_window_from_payload(daily_payload),
            monthly_window=_window_from_payload(monthly_payload),
            reserved_at=datetime.fromisoformat(str(payload["reserved_at"])),
            state=WorkerExecutionReservationState(str(payload["state"])),
            execution_id=(
                ExecutionId(str(execution_id_raw))
                if execution_id_raw is not None
                else None
            ),
            bound_at=(
                datetime.fromisoformat(str(bound_at_raw))
                if bound_at_raw is not None
                else None
            ),
            released_at=(
                datetime.fromisoformat(str(released_at_raw))
                if released_at_raw is not None
                else None
            ),
        )
