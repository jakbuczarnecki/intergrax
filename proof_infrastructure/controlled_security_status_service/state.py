# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorV1,
    SecurityStatusResponseV1,
)


class SecurityStatusStore:
    """Deterministic in-memory security status authority for proof runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._projects: dict[str, SecurityStatusResponseV1] = {}
        self._read_request_count = 0
        self._read_behavior = SecurityStatusReadBehaviorV1.NORMAL

    def put_security(self, status: SecurityStatusResponseV1) -> None:
        with self._lock:
            self._projects[status.project_id] = status

    def get_security(self, project_id: str) -> SecurityStatusResponseV1 | None:
        with self._lock:
            return self._projects.get(project_id)

    def read_security(self, project_id: str) -> SecurityStatusResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            return self._projects.get(project_id)

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0

    def set_read_behavior(self, behavior: SecurityStatusReadBehaviorV1) -> None:
        with self._lock:
            self._read_behavior = behavior

    def read_behavior(self) -> SecurityStatusReadBehaviorV1:
        with self._lock:
            return self._read_behavior
