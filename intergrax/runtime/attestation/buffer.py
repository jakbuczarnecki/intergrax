# © Artur Czarnecki. All rights reserved.

"""In-memory per-run buffer for boundary events surfaced in Tier-3 API responses."""

from __future__ import annotations

import threading
from typing import Sequence

from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1


class BoundaryEventBuffer:
    """Thread-safe store keyed by ``run_id`` (PoC response sink)."""

    def __init__(self) -> None:
        self._rows: dict[str, list[ExecutionBoundaryEventV1]] = {}
        self._sequences: dict[str, int] = {}
        self._lock = threading.Lock()

    def append(self, run_id: str, event: ExecutionBoundaryEventV1) -> ExecutionBoundaryEventV1:
        if not run_id:
            return event
        with self._lock:
            sequence = self._sequences.get(run_id, 0) + 1
            self._sequences[run_id] = sequence
            stored = event.model_copy(update={"event_sequence": sequence})
            self._rows.setdefault(run_id, []).append(stored)
            return stored

    def list_for_run(self, run_id: str) -> list[ExecutionBoundaryEventV1]:
        with self._lock:
            return list(self._rows.get(run_id, ()))

    def snapshot_for_run(self, run_id: str) -> list[dict[str, object]]:
        events = sorted(
            self.list_for_run(run_id),
            key=lambda event: event.event_sequence,
        )
        return [event.model_dump(mode="json") for event in events]

    def all_run_ids(self) -> Sequence[str]:
        with self._lock:
            return tuple(self._rows.keys())
