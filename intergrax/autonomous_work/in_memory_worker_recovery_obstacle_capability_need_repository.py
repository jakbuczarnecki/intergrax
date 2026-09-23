# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory durable obstacle capability need store (AW tests and local composition)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.capability_acquisition import WorkerCapabilityNeed
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId


class InMemoryWorkerRecoveryObstacleCapabilityNeedRepository:
    """Authoritative in-memory store keyed by worker instance and obstacle identity."""

    def __init__(self) -> None:
        self._needs: dict[tuple[str, str], WorkerCapabilityNeed] = {}

    def record_obstacle_capability_need(self, need: WorkerCapabilityNeed) -> None:
        key = (str(need.worker_instance_id), need.obstacle_id)
        self._needs[key] = need

    def get_obstacle_capability_need(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        obstacle_id: str,
    ) -> WorkerCapabilityNeed | None:
        return self._needs.get((str(worker_instance_id), obstacle_id))


__all__ = ["InMemoryWorkerRecoveryObstacleCapabilityNeedRepository"]
