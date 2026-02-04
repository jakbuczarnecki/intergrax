# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import atexit
import threading
import time
import weakref
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from typing import Optional

from intergrax.fastapi_core.execution.adapter import CancellableExecutionAdapter
from intergrax.fastapi_core.execution.capabilities import ExecutionCapabilities
from intergrax.fastapi_core.execution.decision_engine import ExecutionDecisionEngine
from intergrax.fastapi_core.execution.decisions import ExecutionDecision
from intergrax.fastapi_core.execution.failure_classifier import FailureClassifier
from intergrax.fastapi_core.execution.failures import FailureCategory
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.policies import ExecutionPolicy
from intergrax.fastapi_core.execution.worker_contract import (
    CancellableExecutionWorker,
)
from intergrax.fastapi_core.runs.models import RunStatus
from intergrax.fastapi_core.runs.service import RunService


class ThreadedExecutionAdapter(CancellableExecutionAdapter):
    """
    Threaded execution adapter responsibilities:
    - lifecycle boundary (RUNNING → COMPLETED / FAILED)
    - cancellation (best-effort)
    - timeout enforcement (watchdog)
    """

    def __init__(
        self,
        worker: CancellableExecutionWorker,
        run_service: RunService,
        policy: Optional[ExecutionPolicy] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        max_workers: int = 4,
    ) -> None:
        
        if not isinstance(worker, CancellableExecutionWorker):
            raise TypeError(
                "ThreadedExecutionAdapter requires a CancellableExecutionWorker."
            )

        self._worker = worker
        self._run_service = run_service
        self._policy = policy or ExecutionPolicy.default()

        self._failure_classifier = FailureClassifier()
        self._decision_engine = ExecutionDecisionEngine()

        self._capabilities = ExecutionCapabilities(
            supports_retry=True,
            supports_timeout=True,
            supports_cancel=True,
        )

        self._lock = threading.Lock()
        self._futures: dict[str, Future[object]] = {}
        self._deadlines: dict[str, float] = {}

        self._shutdown = False

        if executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="intergrax-exec",
            )
            self._owns_executor = True
        else:
            self._executor = executor
            self._owns_executor = False

        # Lifespan shutdown in FastAPI is the primary mechanism.
        # Tests may forget to close TestClient, so we add a deterministic last-resort cleanup.
        self._atexit_handler = None
        if self._owns_executor:
            self_ref = weakref.ref(self)            

            def _on_exit() -> None:
                obj = self_ref()
                if obj is None:
                    return
                obj.shutdown(wait=False)
            
            atexit.register(_on_exit)
            self._atexit_handler = _on_exit

        # Safety-net: if tests/clients forget to call shutdown(), do not leak threads.
        # This will run deterministically in CPython when adapter is dereferenced
        # (e.g., after a pytest fixture scope ends).
        self._finalizer = weakref.finalize(
            self,
            ThreadedExecutionAdapter._finalize_resources,
            self._executor,
            self._owns_executor,
        )

        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="intergrax-execution-watchdog",
            daemon=True,
        )
        self._watchdog.start()

    @staticmethod
    def _finalize_resources(executor: ThreadPoolExecutor, owns_executor: bool) -> None:
        if not owns_executor:
            return
        try:
            executor.shutdown(wait=False)
        except Exception:
            # Finalizer must never raise.
            pass

    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> ExecutionCapabilities:
        return self._capabilities

    async def start_execution(self, request: ExecutionRequest) -> None:                
        run_id = request.run_id

        # Adapter boundary: expects run already exists (PENDING) and moves it to RUNNING.
        self._run_service.mark_running(run_id)

        def _run() -> None:
            attempt = 0

            try:
                while True:
                    if not self._is_running(run_id):
                        return

                    try:
                        result_payload = self._worker.execute(request)

                        if not self._is_running(run_id):
                            return

                        self._run_service.mark_completed(
                            run_id,
                            result_payload=result_payload,
                        )
                        return

                    except CancelledError:
                        return

                    except Exception as exc:
                        category = self._failure_classifier.classify(exc)
                        decision = self._decision_engine.decide(
                            category=category,
                            attempt=attempt,
                            policy=self._policy,
                        )

                        if decision == ExecutionDecision.RETRY:
                            attempt += 1
                            continue

                        if decision == ExecutionDecision.IGNORE:
                            return

                        if self._is_running(run_id):
                            self._run_service.mark_failed(
                                run_id=run_id,
                                error_type=category.value,
                                error_message=str(exc),
                            )
                        return

            except BaseException as exc:
                # Safety net: never raise from worker thread boundary
                try:
                    if self._is_running(run_id):
                        self._run_service.mark_failed(
                            run_id=run_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                except Exception:
                    pass
            
            finally:
                should_shutdown_executor = False
                with self._lock:
                    self._futures.pop(run_id, None)
                    self._deadlines.pop(run_id, None)

                    # If we own the executor and there is no more work, we can shut it down.
                    if self._owns_executor and not self._shutdown and not self._futures:
                        should_shutdown_executor = True

                if should_shutdown_executor:
                    try:
                        self._executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass


        future = self._executor.submit(_run)

        with self._lock:
            self._futures[run_id] = future
            self._deadlines[run_id] = time.monotonic() + float(self._policy.timeout_seconds)


    # ------------------------------------------------------------------

    def cancel_execution(self, run_id: str) -> None:
        with self._lock:
            future = self._futures.get(run_id)

        if future is not None:
            future.cancel()

        if isinstance(self._worker, CancellableExecutionWorker):
            self._worker.cancel(run_id)

    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        while True:
            # Stop if adapter explicitly shut down
            if self._shutdown:
                return

            now = time.monotonic()

            with self._lock:
                items = list(self._deadlines.items())
                has_work = bool(self._futures)

            for run_id, deadline in items:
                if now > deadline:
                    self._handle_timeout(run_id)
            
            time.sleep(0.1)


    def _handle_timeout(self, run_id: str) -> None:
        """
        Timeout is terminal, but must be idempotent:
        - do not raise
        - do not attempt illegal transitions
        """
        with self._lock:
            future = self._futures.pop(run_id, None)
            self._deadlines.pop(run_id, None)

        if future is not None:
            future.cancel()

        if isinstance(self._worker, CancellableExecutionWorker):
            self._worker.cancel(run_id)
        elif hasattr(self._worker, "stop"):
            self._worker.stop()        

        try:
            if self._is_running(run_id):
                self._run_service.mark_failed(
                    run_id=run_id,
                    error_type=FailureCategory.TIMEOUT.value,
                    error_message="Execution exceeded allowed time",
                )
        except Exception:
            # Watchdog must never crash the process / tests.
            pass

    def _is_running(self, run_id: str) -> bool:
        try:
            return self._run_service.get_run(run_id).status == RunStatus.RUNNING
        except Exception:
            return False

    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """
        Graceful shutdown. Must be idempotent and must NOT raise.
        """

        # attempt cooperative worker stop
        try:
            if isinstance(self._worker, CancellableExecutionWorker):
                for run_id in list(self._futures.keys()):
                    self._worker.cancel(run_id)
            elif hasattr(self._worker, "stop"):
                self._worker.stop()
        except Exception:
            pass

        try:
            self._shutdown = True

            if self._owns_executor:
                self._executor.shutdown(wait=wait, cancel_futures=True)

            with self._lock:
                self._futures.clear()
                self._deadlines.clear()

            # We shut down explicitly -> do not run finalizer later.
            try:
                self._finalizer.detach()
            except Exception:
                pass

            try:
                if self._atexit_handler is not None:
                    atexit.unregister(self._atexit_handler)
                    self._atexit_handler = None
            except Exception:
                pass

        except Exception:
            # Contract: shutdown must not raise.
            pass

