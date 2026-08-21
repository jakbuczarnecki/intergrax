# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading

from proof_infrastructure.controlled_change_approval_service.models import (
    ChangeApprovalResponseV1,
)


class ChangeApprovalStore:
    """Deterministic in-memory change approval authority for proof runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changes: dict[str, ChangeApprovalResponseV1] = {}
        self._read_request_count = 0

    def put_change(self, status: ChangeApprovalResponseV1) -> None:
        with self._lock:
            self._changes[status.change_id] = status

    def get_change(self, change_id: str) -> ChangeApprovalResponseV1 | None:
        with self._lock:
            return self._changes.get(change_id)

    def read_change(self, change_id: str) -> ChangeApprovalResponseV1 | None:
        with self._lock:
            self._read_request_count += 1
            return self._changes.get(change_id)

    def read_request_count(self) -> int:
        with self._lock:
            return self._read_request_count

    def reset_read_request_count(self) -> None:
        with self._lock:
            self._read_request_count = 0
