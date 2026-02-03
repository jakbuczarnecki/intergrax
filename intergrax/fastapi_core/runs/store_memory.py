# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
from typing import Dict

from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from datetime import datetime


class InMemoryRunStore(RunStore):
    """
    In-memory RunStore implementation.

    Intended for:
    - development
    - tests
    - dry-run mode
    """

    def __init__(self) -> None:
        self._runs: Dict[str, RunResponse] = {}

    def create(self) -> RunResponse:
        run_id = uuid.uuid4().hex
        run = RunResponse(run_id=run_id, status=RunStatus.PENDING)
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunResponse:
        return self._runs[run_id]

    def cancel(self, run_id: str) -> RunResponse:
        run = self._runs[run_id]
        canceled = RunResponse(run_id=run.run_id, status=RunStatus.CANCELED)
        self._runs[run_id] = canceled
        return canceled
    
    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        duration_ms: int | None = None,
        result_payload: dict | None = None,
    ) -> RunResponse:
        run = self._runs[run_id]

        updated = RunResponse(
            run_id=run.run_id,
            status=status,
            error_type=error_type if error_type is not None else run.error_type,
            error_message=error_message if error_message is not None else run.error_message,
            started_at=started_at if started_at is not None else run.started_at,
            finished_at=finished_at if finished_at is not None else run.finished_at,
            duration_ms=duration_ms if duration_ms is not None else run.duration_ms,
            result_payload=result_payload if result_payload is not None else run.result_payload,
        )

        self._runs[run_id] = updated
        return updated



