from fastapi import BackgroundTasks
import pytest
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus
from intergrax.fastapi_core.runs.state_machine import InvalidRunTransitionError
from tests._support.builder import DummyRunStore


# class DummyRunStore(RunStore):
#     def __init__(self) -> None:
#         self._runs: dict[str, RunResponse] = {}

#     def create(self) -> RunResponse:
#         run = RunResponse(run_id="r1", status=RunStatus.PENDING)
#         self._runs["r1"] = run
#         return run

#     def get(self, run_id: str) -> RunResponse:
#         return self._runs[run_id]

#     def cancel(self, run_id: str) -> RunResponse:
#         raise AssertionError("Should not reach store.cancel() if transition invalid")

#     def update_status(
#         self,
#         run_id: str,
#         status: RunStatus,
#         *,
#         error_type: str | None = None,
#         error_message: str | None = None,
#         started_at: datetime | None = None,
#         finished_at: datetime | None = None,
#         duration_ms: int | None = None,
#         result_payload: dict | None = None,
#     ) -> RunResponse:
#         current = self._runs[run_id]

#         updated = RunResponse(
#             run_id=current.run_id,
#             status=status,
#             error_type=error_type or current.error_type,
#             error_message=error_message or current.error_message,
#             started_at=started_at or current.started_at,
#             finished_at=finished_at or current.finished_at,
#             duration_ms=duration_ms or current.duration_ms,
#             result_payload=result_payload or current.result_payload,
#         )

#         self._runs[run_id] = updated
#         return updated


class NoOpExecutionAdapter:
    async def start_execution(self, request):
        return

def test_cancel_completed_run_is_blocked_by_state_machine() -> None:
    store = DummyRunStore()
    service = DefaultRunService(store, execution_adapter=NoOpExecutionAdapter())

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
