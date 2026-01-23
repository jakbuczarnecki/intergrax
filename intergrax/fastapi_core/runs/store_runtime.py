# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict

from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.logging import IntergraxLogging
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


class RuntimeRunStore(RunStore):
    """
    RunStore implementation backed by Intergrax runtime (dry-run).

    Responsibilities:
    - Allocate run_id via runtime
    - Track status (PENDING only in dry-run)
    - Provide stable integration point for future execution
    """

    def __init__(self) -> None:
        self._runs: Dict[str, RunResponse] = {}
        self._states: Dict[str, RuntimeState] = {}
        self._logger = IntergraxLogging.get_logger(__name__, component="api")


    def create(self) -> RunResponse:
        """
        Create a new run via runtime in dry-run mode.
        """
        # In dry-run we only allocate run_id and emit trace
        run_id = self._allocate_run_id()

        state = RuntimeState()

        run = RunResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
        )
        self._runs[run_id] = run

        self._logger.info(
            "Run created (dry-run)",
            extra={
                "run_id": run_id,
                "mode": "dry-run",
            },
        )

        return run


    def get(self, run_id: str) -> RunResponse:
        return self._runs[run_id]


    def cancel(self, run_id: str) -> RunResponse:
        run = self._runs[run_id]

        canceled = RunResponse(
            run_id=run.run_id,
            status=RunStatus.CANCELED,
        )
        self._runs[run_id] = canceled

        self._logger.info(
            "Run canceled (dry-run)",
            extra={
                "run_id": run_id,
                "mode": "dry-run",
            },
        )

        return canceled

    # ------------------------------------------------------------------

    def _allocate_run_id(self) -> str:
        """
        Allocate run_id.

        In next iteration this will call:
        - RuntimeState.create_run(...)
        - Trace store
        - Budget initialization
        """
        # Temporary: runtime-style deterministic id
        from uuid import uuid4

        return uuid4().hex
