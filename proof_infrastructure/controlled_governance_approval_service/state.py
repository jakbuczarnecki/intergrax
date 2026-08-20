# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

from proof_infrastructure.controlled_governance_approval_service.models import (
    GovernanceApprovalResponseV1,
)


class GovernanceApprovalStore:
    """Deterministic in-memory governance approval authority for proof runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subjects: dict[str, GovernanceApprovalResponseV1] = {}
        self._read_request_count = 0

    def put_governance(self, status: GovernanceApprovalResponseV1) -> None:
        with self._lock:
            self._subjects[status.subject_id] = status

    def get_governance(self, subject_id: str) -> GovernanceApprovalResponseV1 | None:
        with self._lock:
            return self._subjects.get(subject_id)

    def read_governance(self, subject_id: str) -> GovernanceApprovalResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            return self._subjects.get(subject_id)

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0
