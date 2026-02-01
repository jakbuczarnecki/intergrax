# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.fastapi_core.runs.service import RunService
from intergrax.fastapi_core.runs.state_machine import RunStateMachine


class DefaultRunService(RunService):
    """
    Default orchestration service for runs.

    Responsibilities at this stage:
    - orchestrate lifecycle
    - trigger background execution
    - delegate ALL state persistence to RunStore
    - enforce lifecycle transitions via RunStateMachine
    """

    def __init__(self, store: RunStore) -> None:
        self._store = store

    def create_run(
        self,
        context: RequestContext,
        background_tasks: BackgroundTasks,
    ) -> RunResponse:
        """
        Entry point for run lifecycle orchestration.

        Flow:
        1) Create run record
        2) Verify initial state
        3) Schedule background execution
        """
        run = self._store.create()

        # Store must return initial PENDING state
        if run.status != RunStatus.PENDING:
            raise RuntimeError("RunStore.create() must return PENDING run")

        background_tasks.add_task(self._execute_run, run.run_id)
        return run


    def _execute_run(self, run_id: str) -> None:
        """
        Background execution simulation.

        IMPORTANT:
        - No direct object mutation
        - All lifecycle changes go through RunStore
        - All transitions are validated by RunStateMachine
        """
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.RUNNING)
        self._store.update_status(run_id, RunStatus.RUNNING)

        try:
            # Placeholder for real execution

            current = self._store.get(run_id)
            RunStateMachine.validate_transition(current.status, RunStatus.COMPLETED)
            self._store.update_status(run_id, RunStatus.COMPLETED)

        except Exception:
            current = self._store.get(run_id)
            RunStateMachine.validate_transition(current.status, RunStatus.FAILED)
            self._store.update_status(run_id, RunStatus.FAILED)

    def get_run(self, run_id: str) -> RunResponse:
        return self._store.get(run_id)

    def cancel_run(self, run_id: str) -> RunResponse:
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.CANCELED)
        return self._store.update_status(run_id, RunStatus.CANCELED)
