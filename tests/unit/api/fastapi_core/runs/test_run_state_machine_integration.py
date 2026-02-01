from fastapi import BackgroundTasks
import pytest

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.state_machine import InvalidRunTransitionError
from intergrax.fastapi_core.runs.store_base import RunStore


class DummyRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, RunResponse] = {}

    def create(self) -> RunResponse:
        run = RunResponse(run_id="r1", status=RunStatus.PENDING)
        self._runs["r1"] = run
        return run

    def get(self, run_id: str) -> RunResponse:
        return self._runs[run_id]

    def cancel(self, run_id: str) -> RunResponse:
        raise AssertionError("Should not reach store.cancel() if transition invalid")

    def update_status(self, run_id: str, status: RunStatus) -> RunResponse:
        run = RunResponse(run_id=run_id, status=status)
        self._runs[run_id] = run
        return run


def test_cancel_completed_run_is_blocked_by_state_machine() -> None:
    store = DummyRunStore()
    service = DefaultRunService(store)

    ctx = RequestContext(
        request_id="req1",
        path="/runs",
        method="POST",
        tenant_id="t1",
        user_id="u1",
        auth=None,
    )

    bg = BackgroundTasks()

    # Create run
    run = service.create_run(ctx, bg)

    # Force it to COMPLETED via store (legal transition path simulated)
    store.update_status(run.run_id, RunStatus.RUNNING)
    store.update_status(run.run_id, RunStatus.COMPLETED)

    # Now attempt illegal transition: COMPLETED → CANCELED
    with pytest.raises(InvalidRunTransitionError):
        service.cancel_run(run.run_id)
