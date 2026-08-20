# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading
from datetime import UTC, datetime

from proof_infrastructure.controlled_project_status_service.models import (
    ProjectStatusControlUpdateV1,
    ProjectStatusReadBehaviorV1,
    ProjectStatusResponseV1,
)


class ProjectStatusStore:
    """Deterministic in-memory project status authority for proof runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._projects: dict[str, ProjectStatusResponseV1] = {}
        self._read_request_count = 0
        self._read_behavior = ProjectStatusReadBehaviorV1.NORMAL

    def reset(self) -> None:
        with self._lock:
            self._projects.clear()
            self._read_request_count = 0
            self._read_behavior = ProjectStatusReadBehaviorV1.NORMAL

    def put_status(self, status: ProjectStatusResponseV1) -> None:
        with self._lock:
            self._projects[status.project_id] = status

    def get_status(self, project_id: str) -> ProjectStatusResponseV1 | None:
        with self._lock:
            return self._projects.get(project_id)

    def read_status(self, project_id: str) -> ProjectStatusResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            return self._projects.get(project_id)

    def update_status(
        self,
        project_id: str,
        update: ProjectStatusControlUpdateV1,
    ) -> ProjectStatusResponseV1:
        with self._lock:
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError("project_not_found")
            updated = current.model_copy(
                update={
                    "readiness_score": (
                        update.readiness_score
                        if update.readiness_score is not None
                        else current.readiness_score
                    ),
                    "blockers": (
                        update.blockers
                        if update.blockers is not None
                        else current.blockers
                    ),
                    "status": (
                        update.status if update.status is not None else current.status
                    ),
                    "updated_at": datetime.now(UTC),
                }
            )
            self._projects[project_id] = updated
            return updated

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0

    def set_read_behavior(self, behavior: ProjectStatusReadBehaviorV1) -> None:
        with self._lock:
            self._read_behavior = behavior

    def read_behavior(self) -> ProjectStatusReadBehaviorV1:
        with self._lock:
            return self._read_behavior
