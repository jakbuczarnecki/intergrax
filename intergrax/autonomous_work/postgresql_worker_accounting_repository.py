# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production PostgreSQL adapter for worker accounting windows (AW-5B)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from intergrax.autonomous_work.in_memory_worker_accounting_repository import (
    InMemoryWorkerAccountingRepository,
)
from intergrax.autonomous_work.repository import AutonomousWorkRepositoryCapabilities
from intergrax.autonomous_work.worker_budget_ports import (
    WorkerAccountingConflict,
    WorkerAccountingNotFound,
)
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    BudgetUsageTotals,
    WorkerAccountingState,
    WorkerAccountingWindow,
    WorkerBudgetAdmissionResult,
    WorkerBudgetReserveRequest,
    WorkerExecutionReservation,
    WorkerLogicalDispatchRef,
    WorkerProactiveEvaluationAccountingRequest,
)
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLSession,
)

if TYPE_CHECKING:
    from intergrax.autonomous_work.postgresql_repository import PostgreSQLAutonomousWorkStore

_BACKEND_ID = "autonomous_work.worker_accounting.postgresql"
_CAPABILITIES = AutonomousWorkRepositoryCapabilities(
    backend_id=_BACKEND_ID,
    durable=True,
    reference_only=False,
)

_ResultT = TypeVar("_ResultT")


class PostgreSQLWorkerAccountingRepository:
    """Durable worker accounting repository backed by CAS snapshots per worker."""

    _SNAPSHOT_TABLE = "aw_worker_accounting_snapshots"

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def get_window_state(
        self,
        *,
        window: WorkerAccountingWindow,
    ) -> WorkerAccountingState | None:
        worker_id = window.worker_instance_id
        with self._store.transaction() as conn:
            repo = self._load_worker_repo(conn, worker_id.strip())
            return repo.get_window_state(window=window)

    def reserve(self, request: WorkerBudgetReserveRequest) -> WorkerBudgetAdmissionResult:
        worker_id = request.logical_dispatch.worker_instance_id.strip()
        return self._mutate_worker(
            worker_id,
            lambda repo: repo.reserve(request),
        )

    def bind_execution(
        self,
        *,
        logical_dispatch: WorkerLogicalDispatchRef,
        execution_id: ExecutionId,
        bound_at: datetime,
    ) -> WorkerExecutionReservation:
        worker_id = logical_dispatch.worker_instance_id.strip()
        return self._mutate_worker(
            worker_id,
            lambda repo: repo.bind_execution(
                logical_dispatch=logical_dispatch,
                execution_id=execution_id,
                bound_at=bound_at,
            ),
        )

    def release_reservation(
        self,
        *,
        logical_dispatch: WorkerLogicalDispatchRef,
        released_at: datetime,
    ) -> WorkerExecutionReservation:
        worker_id = logical_dispatch.worker_instance_id.strip()
        return self._mutate_worker(
            worker_id,
            lambda repo: repo.release_reservation(
                logical_dispatch=logical_dispatch,
                released_at=released_at,
            ),
        )

    def release_execution(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        released_at: datetime,
    ) -> WorkerExecutionReservation:
        return self._mutate_worker(
            worker_instance_id.strip(),
            lambda repo: repo.release_execution(
                worker_instance_id=worker_instance_id,
                execution_id=execution_id,
                released_at=released_at,
            ),
        )

    def record_consumption(
        self,
        *,
        worker_instance_id: str,
        execution_id: ExecutionId,
        usage: BudgetUsageTotals,
        recorded_at: datetime,
    ) -> None:
        self._mutate_worker(
            worker_instance_id.strip(),
            lambda repo: repo.record_consumption(
                worker_instance_id=worker_instance_id,
                execution_id=execution_id,
                usage=usage,
                recorded_at=recorded_at,
            ),
        )

    def record_proactive_evaluation(
        self,
        request: WorkerProactiveEvaluationAccountingRequest,
    ) -> WorkerBudgetAdmissionResult:
        worker_id = request.worker_instance_id.strip()
        return self._mutate_worker(
            worker_id,
            lambda repo: repo.record_proactive_evaluation(request),
        )

    def _mutate_worker(
        self,
        worker_instance_id: str,
        operation: Callable[[InMemoryWorkerAccountingRepository], _ResultT],
    ) -> _ResultT:
        with self._store.transaction() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (worker_instance_id,),
            )
            revision, repo = self._load_worker_repo_with_revision(conn, worker_instance_id)
            result = operation(repo)
            snapshot_json = json.dumps(repo.to_snapshot())
            if revision is None:
                conn.execute(
                    f"""
                    INSERT INTO {self._SNAPSHOT_TABLE} (
                        worker_instance_id, snapshot_json, revision
                    ) VALUES (%s, %s, %s)
                    """,
                    (worker_instance_id, snapshot_json, 1),
                )
            else:
                updated = conn.execute(
                    f"""
                    UPDATE {self._SNAPSHOT_TABLE}
                    SET snapshot_json = %s, revision = revision + 1
                    WHERE worker_instance_id = %s AND revision = %s
                    """,
                    (snapshot_json, worker_instance_id, revision),
                )
                if updated.rowcount != 1:
                    raise WorkerAccountingConflict(
                        "worker accounting snapshot revision conflict"
                    )
            return result

    def _load_worker_repo(
        self,
        conn: PostgreSQLSession,
        worker_instance_id: str,
    ) -> InMemoryWorkerAccountingRepository:
        _revision, repo = self._load_worker_repo_with_revision(conn, worker_instance_id)
        return repo

    def _load_worker_repo_with_revision(
        self,
        conn: PostgreSQLSession,
        worker_instance_id: str,
    ) -> tuple[int | None, InMemoryWorkerAccountingRepository]:
        row = conn.execute(
            f"""
            SELECT snapshot_json, revision
            FROM {self._SNAPSHOT_TABLE}
            WHERE worker_instance_id = %s
            """,
            (worker_instance_id,),
        ).fetchone()
        if row is None:
            return None, InMemoryWorkerAccountingRepository()
        payload = json.loads(row["snapshot_json"])
        if not isinstance(payload, dict):
            raise WorkerAccountingNotFound("corrupt worker accounting snapshot")
        return int(row["revision"]), InMemoryWorkerAccountingRepository.from_snapshot(payload)
