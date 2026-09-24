# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL durable store for worker obstacle capability needs (UCA-6C-R6-R5.9)."""

from __future__ import annotations

from intergrax.autonomous_work.postgresql_repository import (
    PostgreSQLAutonomousWorkStore,
    _unique_violation,
)
from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityConflict,
    AutonomousWorkRepositoryCapabilities,
)
from intergrax.autonomous_work.worker_capability_need_serialization import (
    worker_capability_need_from_json,
    worker_capability_need_to_json,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId

_TABLE = "aw_worker_obstacle_capability_needs"

_CAPABILITIES = AutonomousWorkRepositoryCapabilities(
    backend_id="autonomous_work.worker_obstacle_capability_need.postgresql",
    durable=True,
    reference_only=False,
)


class PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository:
    """Durable AW-owned obstacle capability need index — immutable insert semantics."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def record_obstacle_capability_need(self, need: WorkerCapabilityNeed) -> None:
        record_json = worker_capability_need_to_json(need)
        worker_id = str(need.worker_instance_id)
        obstacle_id = need.obstacle_id
        try:
            with self._store.transaction() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {_TABLE} (
                        worker_instance_id,
                        obstacle_id,
                        record_json
                    ) VALUES (%s, %s, %s)
                    """,
                    (worker_id, obstacle_id, record_json),
                )
        except Exception as exc:
            if not _unique_violation(exc):
                raise
            existing = self.get_obstacle_capability_need(
                worker_instance_id=need.worker_instance_id,
                obstacle_id=obstacle_id,
            )
            if existing is None:
                raise AutonomousWorkEntityConflict(
                    "worker obstacle capability need already exists with different content "
                    f"for {worker_id}:{obstacle_id}",
                ) from exc
            if existing == need:
                return
            raise AutonomousWorkEntityConflict(
                "worker obstacle capability need already exists with different content "
                f"for {worker_id}:{obstacle_id}",
            ) from exc

    def get_obstacle_capability_need(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        obstacle_id: str,
    ) -> WorkerCapabilityNeed | None:
        with self._store.transaction() as conn:
            row = conn.execute(
                f"""
                SELECT record_json FROM {_TABLE}
                WHERE worker_instance_id = %s AND obstacle_id = %s
                """,
                (str(worker_instance_id), obstacle_id),
            ).fetchone()
        if row is None:
            return None
        return worker_capability_need_from_json(row["record_json"])


__all__ = ["PostgreSQLWorkerRecoveryObstacleCapabilityNeedRepository"]
