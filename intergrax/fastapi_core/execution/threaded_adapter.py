# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from intergrax.fastapi_core.execution.adapter import CancellableExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.worker_contract import (
    CancellableExecutionWorker,
    ExecutionWorker,
)
from intergrax.fastapi_core.runs.service import RunService


class ThreadedExecutionAdapter(CancellableExecutionAdapter):
    """
    Threaded execution adapter with governance responsibilities:
    - lifecycle boundary (RUNNING → COMPLETED / FAILED)
    - cancellation
    - timeout enforcement (watchdog)
    """

    def __init__(
        self,
        worker: ExecutionWorker,
        run_service: RunService,
        executor: Optional[ThreadPoolExecutor] = None,
        max_workers: int = 4,
        timeout_seconds: int = 300,
    ) -> None:
        self._worker = worker
        self._run_service = run_service
        self._timeout_seconds = timeout_seconds

        self._futures: dict[str, Future] = {}
        self._deadlines: dict[str, float] = {}
        self._shutdown = False

        if executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._owns_executor = True
        else:
            self._executor = executor
            self._owns_executor = False

        # Start watchdog thread
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="intergrax-execution-watchdog",
            daemon=True,
        )
        self._watchdog.start()

    # ------------------------------------------------------------------

    async def start_execution(self, request: ExecutionRequest) -> None:
        run_id = request.run_id

        # Lifecycle boundary
        self._run_service.mark_running(run_id)

        def _run() -> None:
            try:
                self._worker.execute(request)
                self._run_service.mark_completed(run_id)
            except Exception as exc:
                self._run_service.mark_failed(
                    run_id=run_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            finally:
                self._futures.pop(run_id, None)
                self._deadlines.pop(run_id, None)

        future = self._executor.submit(_run)

        self._futures[run_id] = future
        self._deadlines[run_id] = time.monotonic() + self._timeout_seconds

    # ------------------------------------------------------------------

    def cancel_execution(self, run_id: str) -> None:
        future = self._futures.get(run_id)
        if future:
            future.cancel()

        if isinstance(self._worker, CancellableExecutionWorker):
            self._worker.cancel(run_id)

    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        while not self._shutdown:
            now = time.monotonic()

            for run_id, deadline in list(self._deadlines.items()):
                if now > deadline:
                    self._handle_timeout(run_id)

            time.sleep(0.1)

    def _handle_timeout(self, run_id: str) -> None:
        future = self._futures.pop(run_id, None)
        self._deadlines.pop(run_id, None)

        if future:
            future.cancel()

        if isinstance(self._worker, CancellableExecutionWorker):
            self._worker.cancel(run_id)

        self._run_service.mark_failed(
            run_id=run_id,
            error_type="TimeoutError",
            error_message="Execution exceeded allowed time",
        )

    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown = True

        if self._owns_executor:
            self._executor.shutdown(wait=wait)
