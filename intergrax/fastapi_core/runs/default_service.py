# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional

from fastapi import BackgroundTasks
from datetime import datetime
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.adapters.adapter import CancellableExecutionAdapter, ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.models import CreateRunRequest, RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.fastapi_core.runs.service import RunService
from intergrax.fastapi_core.runs.state_machine import RunStateMachine
from intergrax.utils.time_provider import SystemTimeProvider


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
        *,
        create_request: Optional[CreateRunRequest] = None,
    ) -> RunResponse:
        run = self._store.create()

        if run.status != RunStatus.PENDING:
            raise RuntimeError("RunStore.create() must return PENDING run")

        request = ExecutionRequest(
            run_id=run.run_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            input_payload=dict(create_request.payload) if create_request else {},
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
        response = self._store.update_status(run_id, RunStatus.CANCELED)

        if isinstance(self._execution_adapter, CancellableExecutionAdapter):
            self._execution_adapter.cancel_execution(run_id)

        return response


    def mark_completed(self, run_id: str, result_payload: Optional[dict] = None) -> None:
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.COMPLETED)

        finished = SystemTimeProvider.utc_now()

        duration = None
        if current.started_at:
            duration = int((finished - current.started_at).total_seconds() * 1000)

        self._store.update_status(
            run_id,
            RunStatus.COMPLETED,
            finished_at=finished,
            duration_ms=duration,
            result_payload=result_payload,
        )


    def mark_failed(
        self,
        run_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.FAILED)

        finished = SystemTimeProvider.utc_now()

        duration = None
        if current.started_at:
            duration = int((finished - current.started_at).total_seconds() * 1000)

        self._store.update_status(
            run_id,
            RunStatus.FAILED,
            error_type=error_type,
            error_message=error_message,
            finished_at=finished,
            duration_ms=duration,
        )


    def mark_running(self, run_id: str) -> None:
        current = self._store.get(run_id)
        RunStateMachine.validate_transition(current.status, RunStatus.RUNNING)
        
        self._store.update_status(
            run_id,
            RunStatus.RUNNING,
            started_at=SystemTimeProvider.utc_now(),
        )