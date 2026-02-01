# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from fastapi import BackgroundTasks
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.fastapi_core.runs.service import RunService


class DefaultRunService(RunService):
    """
    Default orchestration service for runs.

    Currently acts as a thin layer over RunStore.
    Future responsibilities:
    - lifecycle transitions
    - background execution
    - retry logic
    - cancellation semantics
    """

    def __init__(self, store: RunStore) -> None:
        self._store = store

    def create_run(self, context: RequestContext, background_tasks: BackgroundTasks) -> RunResponse:
        """
        Entry point for run lifecycle orchestration.

        Current behavior:
        - delegates run creation to persistence layer

        Future responsibilities:
        - initialize lifecycle state
        - transition to RUNNING
        - spawn execution (background)
        - handle failures / retries
        """
        run = self._store.create()

        # Lifecycle orchestration hook (reserved for future logic)
        # Example future:
        # self._transition_to_running(run, context)

        # run.status = RunStatus.PENDING
        run = self._store.update_status(run.run_id, RunStatus.PENDING)

        # run = self.start_run(run_id=run.run_id)
        background_tasks.add_task(self._execute_run, run.run_id)

        return run
    
    def _execute_run(self, run_id: str) -> None:
        run = self._store.get(run_id)
        
        run.status = RunStatus.RUNNING

        try:
            run.status = RunStatus.COMPLETED
        except Exception:
            run.status = RunStatus.FAILED



    def get_run(self, run_id: str) -> RunResponse:
        return self._store.get(run_id)


    def cancel_run(self, run_id: str) -> RunResponse:
        return self._store.cancel(run_id)

