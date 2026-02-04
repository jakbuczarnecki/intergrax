import time
from threading import Event

from intergrax.fastapi_core.execution.worker_contract import CancellableExecutionWorker
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.adapters.threaded_adapter import ThreadedExecutionAdapter
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus
from tests._support.builder import DummyRunStore


class BlockingWorker(CancellableExecutionWorker):
    def __init__(self) -> None:
        self._stop = Event()

    def execute(self, request: ExecutionRequest) -> None:
        # simulate long running job
        while not self._stop.is_set():
            time.sleep(0.01)

    def stop(self) -> None:
        self._stop.set()

    def cancel(self, run_id: str) -> None:
        self._stop.set()


def test_cancel_running_threaded_execution() -> None:
    store = DummyRunStore()
    service = DefaultRunService(store=store, execution_adapter=None)

    worker = BlockingWorker()
    adapter = ThreadedExecutionAdapter(worker=worker, run_service=service)
    service._execution_adapter = adapter

    run = store.create()

    request = ExecutionRequest(
        run_id=run.run_id,
        tenant_id="t",
        user_id="u",
        input_payload={},
        metadata={},
    )

    # start execution
    import asyncio
    asyncio.run(adapter.start_execution(request))

    # cancel
    service.cancel_run(run.run_id)

    # give thread time
    time.sleep(0.1)

    assert store.get(run.run_id).status == RunStatus.CANCELED
