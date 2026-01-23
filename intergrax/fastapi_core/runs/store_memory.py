# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
from typing import Dict

from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore



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
