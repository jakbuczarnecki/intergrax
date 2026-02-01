# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.fastapi_core.runs.models import RunStatus
from intergrax.fastapi_core.runs.state_machine import RunStateMachine


class ExecutionWorker:
    """
    Minimal execution worker contract.

    A worker:
    - receives ExecutionRequest
    - owns execution lifecycle
    - MUST validate all status transitions via RunStateMachine
    """

    def __init__(self, store: RunStore) -> None:
        self._store = store

    def execute(self, request: ExecutionRequest) -> None:
        run_id = request.run_id

        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.RUNNING)
        self._store.update_status(run_id, RunStatus.RUNNING)

        try:
            # --- real execution goes here ---
            # use request.input_payload / request.config

            current = self._store.get(run_id)
            RunStateMachine.validate_transition(current.status, RunStatus.COMPLETED)
            self._store.update_status(run_id, RunStatus.COMPLETED)

        except Exception:
            current = self._store.get(run_id)
            RunStateMachine.validate_transition(current.status, RunStatus.FAILED)
            self._store.update_status(run_id, RunStatus.FAILED)
            raise
