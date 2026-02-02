# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
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

    def __init__(
        self,
        store: RunStore,
        execution_adapter: ExecutionAdapter,
    ) -> None:
        self._store = store
        self._execution_adapter = execution_adapter


    def create_run(
        self,
        context: RequestContext,
        background_tasks: BackgroundTasks,
    ) -> RunResponse:
        run = self._store.create()

        if run.status != RunStatus.PENDING:
            raise RuntimeError("RunStore.create() must return PENDING run")

        request = ExecutionRequest(
            run_id=run.run_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            input_payload={},
            metadata={"request_id": context.request_id},
        )

        background_tasks.add_task(
            self._execution_adapter.start_execution,
            request,
        )

        return run


    def get_run(self, run_id: str) -> RunResponse:
        return self._store.get(run_id)


    def cancel_run(self, run_id: str) -> RunResponse:
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.CANCELED)
        return self._store.update_status(run_id, RunStatus.CANCELED)


    def mark_completed(self, run_id: str) -> None:
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.COMPLETED)
        self._store.update_status(run_id, RunStatus.COMPLETED)


    def mark_failed(
        self,
        run_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        current = self._store.get(run_id)

        RunStateMachine.validate_transition(current.status, RunStatus.FAILED)

        # P0: error info stored directly on run model
        self._store.update_status(
            run_id,
            RunStatus.FAILED,
            error_type=error_type,
            error_message=error_message,
        )


    def mark_running(self, run_id: str) -> None:
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.RUNNING)
        self._store.update_status(run_id, RunStatus.RUNNING)
