# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker execution accounting lifecycle helpers (AW-5B)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from intergrax.autonomous_work.worker_budget_ports import (
    ExecutionUsageProvider,
    WorkerAccountingConflict,
    WorkerAccountingNotFound,
    WorkerAccountingRepository,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionDispatchRequest,
)
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    WorkerLogicalDispatchRef,
)
from intergrax.contracts.execution_identity import ExecutionId


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WorkerExecutionAccountingService:
    """Bind, release, and record canonical execution usage for worker accounting."""

    def __init__(
        self,
        *,
        accounting_repository: WorkerAccountingRepository,
        usage_provider: ExecutionUsageProvider | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._accounting_repository = accounting_repository
        self._usage_provider = usage_provider
        self._clock = clock or _utc_now

    def logical_dispatch_from_request(
        self,
        request: WorkerExecutionDispatchRequest[object, object],
    ) -> WorkerLogicalDispatchRef:
        return WorkerLogicalDispatchRef(
            worker_instance_id=request.worker_instance_id,
            source_kind=request.source.source_kind,
            source_ref=request.source.source_ref,
        )

    def bind_execution(
        self,
        *,
        request: WorkerExecutionDispatchRequest[object, object],
        execution_id: ExecutionId,
        bound_at: datetime | None = None,
    ) -> None:
        self._accounting_repository.bind_execution(
            logical_dispatch=self.logical_dispatch_from_request(request),
            execution_id=execution_id,
            bound_at=bound_at or self._clock(),
        )

    def release_reservation(
        self,
        *,
        request: WorkerExecutionDispatchRequest[object, object],
        released_at: datetime | None = None,
    ) -> None:
        self._accounting_repository.release_reservation(
            logical_dispatch=self.logical_dispatch_from_request(request),
            released_at=released_at or self._clock(),
        )

    def release_execution(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        released_at: datetime | None = None,
    ) -> None:
        self._accounting_repository.release_execution(
            worker_instance_id=worker_instance_id,
            execution_id=execution_id,
            released_at=released_at or self._clock(),
        )

    def record_terminal_execution(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        released_at: datetime | None = None,
    ) -> None:
        resolved_at = released_at or self._clock()
        self.release_execution(
            worker_instance_id=worker_instance_id,
            execution_id=execution_id,
            released_at=resolved_at,
        )
        if self._usage_provider is None:
            return
        usage = self._usage_provider.get_final_usage(execution_id)
        if usage is None:
            return
        self._accounting_repository.record_consumption(
            worker_instance_id=worker_instance_id,
            execution_id=execution_id,
            usage=usage,
            recorded_at=resolved_at,
        )

    def record_consumption(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        recorded_at: datetime | None = None,
    ) -> None:
        if self._usage_provider is None:
            raise WorkerAccountingNotFound(
                "usage provider is required for canonical consumption recording"
            )
        usage = self._usage_provider.get_final_usage(execution_id)
        if usage is None:
            raise WorkerAccountingNotFound(
                f"final usage unavailable for execution {execution_id}"
            )
        self._accounting_repository.record_consumption(
            worker_instance_id=worker_instance_id,
            execution_id=execution_id,
            usage=usage,
            recorded_at=recorded_at or self._clock(),
        )


__all__ = [
    "WorkerAccountingConflict",
    "WorkerAccountingNotFound",
    "WorkerExecutionAccountingService",
]
