# © Artur Czarnecki. All rights reserved.

"""In-memory per-run buffer for boundary events surfaced in Tier-3 API responses."""

from __future__ import annotations

import threading
from typing import Sequence

from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1


class BoundaryEventBuffer:
    """Thread-safe store keyed by ``run_id`` (PoC v1 response sink)."""

    def __init__(self) -> None:
        self._rows: dict[str, list[ExecutionBoundaryEventV1]] = {}
        self._lock = threading.Lock()

    def append(self, run_id: str, event: ExecutionBoundaryEventV1) -> None:
        if not run_id:
            return
        with self._lock:
            self._rows.setdefault(run_id, []).append(event)

    def list_for_run(self, run_id: str) -> list[ExecutionBoundaryEventV1]:
        with self._lock:
            return list(self._rows.get(run_id, ()))

    def snapshot_for_run(self, run_id: str) -> list[dict[str, object]]:
        return [event.model_dump(mode="json") for event in self.list_for_run(run_id)]

    def all_run_ids(self) -> Sequence[str]:
        with self._lock:
            return tuple(self._rows.keys())
