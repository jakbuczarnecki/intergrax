# © Artur Czarnecki. All rights reserved.

"""Co-located worker that drains DocumentStoreTaskQueue via TaskExecutionRegistry."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
)
from intergrax.queueing.worker.execution import (
    IdempotencyLockConflictError,
    RetryableHandlerError,
    execute_logical_task,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.runtime.background_execution.identity_persistence import (
    BackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryDisposition,
    admit_background_execution_reentry,
)
from intergrax.runtime.background_execution.required_audit_evidence import (
    admit_background_execution_handler,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.execution.attempt_lifecycle.service import AttemptLifecycleService
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.execution.identity_authority import BackgroundTransportIdentity
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)

logger = logging.getLogger(__name__)

InterruptedHandler = Callable[[TaskHandle, TaskRequest], None]


class DocumentStoreTaskWorkerStillRunning(RuntimeError):
    """``start()`` refused: the consumer of an earlier run has not ended after ``stop()`` (EX-L-06)."""


class DocumentStoreTaskWorker:
    """Platform consumer for durable DocumentStore-backed MessageBus tasks."""

    def __init__(
        self,
        queue: DocumentStoreTaskQueue,
        registry: TaskExecutionRegistry,
        *,
        poll_interval_seconds: float = 0.25,
        claim_limit: int = 4,
        on_interrupted: InterruptedHandler | None = None,
        identity_persistence: BackgroundExecutionIdentityPersistence,
        causal_evidence_persistence: CausalEvidencePersistence,
        attempt_lifecycle: AttemptLifecycleService,
        execution_terminal: ExecutionTerminalService,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if retry_policy is not None:
            retry_policy.validate()
        self._retry_policy = retry_policy
        self._queue = queue
        self._registry = registry
        self._poll_interval_seconds = poll_interval_seconds
        self._claim_limit = claim_limit
        self._on_interrupted = on_interrupted
        self._identity_persistence = identity_persistence
        self._causal_evidence_persistence = causal_evidence_persistence
        self._attempt_lifecycle = attempt_lifecycle
        self._execution_terminal = execution_terminal
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._in_flight = 0
        self._idle = threading.Condition()

    def start(self) -> None:
        """Recover interrupted rows, then start one consumer thread.

        A no-op while the current consumer runs. Raises ``DocumentStoreTaskWorkerStillRunning`` when a
        previous ``stop()`` timed out and its consumer is still inside a handler (EX-L-06): starting
        another consumer next to it would run two consumers over one queue.
        """
        thread = self._thread
        if thread is not None and thread.is_alive():
            if not self._stop.is_set():
                return
            raise DocumentStoreTaskWorkerStillRunning(
                "document_store_task_worker_still_running: the previous consumer has not ended after stop()",
            )
        self._thread = None
        recovery_budget = self._retry_policy.max_retries if self._retry_policy is not None else None
        for handle, request in self._queue.recover_interrupted_running(
            max_retries=recovery_budget,
        ):
            if self._on_interrupted is not None:
                try:
                    self._on_interrupted(handle, request)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "document_store_task_worker_interrupt_hook_failed task_id=%s",
                        handle.task_id,
                    )
        # A fresh event per run: a late consumer of an earlier run can never observe a cleared event.
        stop_event = threading.Event()
        self._stop = stop_event
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(stop_event,),
            name="intergrax-document-store-task-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 10.0) -> bool:
        """Close intake and join the consumer; True when it has ended.

        When the join times out the consumer is still referenced (False is returned), so a later
        ``start()`` refuses instead of starting a second consumer next to it.
        """
        self.stop_intake()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_seconds)
        if thread is not None and thread.is_alive():
            return False
        self._thread = None
        return True

    def stop_intake(self) -> None:
        """Stop claiming new rows without joining; rows already claimed are still handled."""
        with self._idle:
            self._stop.set()

    def in_flight_count(self) -> int:
        """Claimed rows not settled yet, plus a claim of the consumer loop that is in progress."""
        with self._idle:
            return self._in_flight

    def wait_idle(self, timeout_seconds: float) -> bool:
        """Wait until no claimed row is being handled; True when idle within the timeout."""
        with self._idle:
            return self._idle.wait_for(lambda: self._in_flight == 0, timeout=timeout_seconds)

    def drain_once(self) -> int:
        claimed = self._queue.claim_pending(limit=self._claim_limit)
        if claimed:
            with self._idle:
                self._in_flight += len(claimed)
        remaining = len(claimed)
        try:
            for handle, request in claimed:
                try:
                    try:
                        transport_ref = BackgroundTransportExecutionRef(
                            tenant_id=request.tenant_id,
                            provider=handle.provider,
                            transport_task_id=handle.task_id,
                        )
                        self._seed_carried_identity(transport_ref, request)
                        reentry = admit_background_execution_reentry(
                            transport_ref=transport_ref,
                            identity_persistence=self._identity_persistence,
                            attempt_lifecycle=self._attempt_lifecycle,
                            execution_terminal=self._execution_terminal,
                        )
                        if reentry.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED:
                            self._queue.mark_succeeded(handle, output=None)
                            continue
                        execution_identity = reentry.identity
                        result = admit_background_execution_handler(
                            transport_ref=transport_ref,
                            execution_identity=execution_identity,
                            causal_evidence_persistence=self._causal_evidence_persistence,
                            handler=lambda: execute_logical_task(
                                registry=self._registry,
                                logical_task_name=request.task_name,
                                tenant_id=execution_identity.tenant_id,
                                run_id=str(execution_identity.run_id),
                                payload=request.payload,
                                idempotency_key=request.idempotency_key,
                                idempotency_store=None,
                                execution_identity=execution_identity,
                            ),
                        )
                        if result.success:
                            output = None
                            if result.output is not None:
                                output = result.output.model_dump_json().encode("utf-8")
                            self._queue.mark_succeeded(handle, output=output)
                        else:
                            message = result.error.error_message if result.error is not None else "handler_failed"
                            self._queue.mark_failed(handle, error_message=message)
                    except Exception as exc:  # noqa: BLE001 - worker plane normalizes failures
                        logger.exception(
                            "document_store_task_worker_failed task_id=%s task_name=%s",
                            handle.task_id,
                            request.task_name,
                        )
                        self._settle_exception(handle, exc)
                finally:
                    remaining -= 1
                    self._settle_in_flight(1)
        finally:
            # Rows not reached because a settlement raised are not handled here any more: never leak them.
            if remaining:
                self._settle_in_flight(remaining)
        return len(claimed)

    def _drain_if_open(self, stop_event: threading.Event) -> int:
        """One loop iteration; intake check and claim slot are taken under one lock (no drain race)."""
        with self._idle:
            if stop_event.is_set():
                return 0
            self._in_flight += 1  # the claim in progress counts as active work
        try:
            return self.drain_once()
        finally:
            self._settle_in_flight()

    def _settle_in_flight(self, count: int = 1) -> None:
        with self._idle:
            self._in_flight -= count
            if self._in_flight == 0:
                self._idle.notify_all()

    def _seed_carried_identity(
        self,
        transport_ref: BackgroundTransportExecutionRef,
        request: TaskRequest,
    ) -> None:
        """Adopt canonical identity carried on the envelope (PDC-D7-R016).

        ``store_if_absent`` keeps the first mapping, so redeliveries and retries of the
        same transport task never re-seed or fork identity. The worker does not mint.
        """
        carried = request.identity
        if carried is None or not carried.has_canonical_execution_ids():
            return
        self._identity_persistence.store_if_absent(
            transport_ref,
            BackgroundTransportIdentity(
                task_id=carried.task_id,
                run_id=carried.run_id,
                attempt_id=carried.attempt_id,
                execution_id=carried.execution_id,
            ),
        )

    def _settle_exception(self, handle: TaskHandle, exc: Exception) -> None:
        """Retryable errors consume the retry budget; exhaustion dead-letters."""
        policy = self._retry_policy
        if policy is not None and _is_retryable(policy, exc):
            self._queue.retry_or_dead_letter(
                handle,
                error_message=str(exc),
                retry_policy=policy,
            )
            return
        self._queue.mark_failed(handle, error_message=str(exc))

    def _run_loop(self, stop_event: threading.Event) -> None:
        delay_raw = (os.environ.get("INTERGRAX_DOCUMENT_STORE_TASK_WORKER_START_DELAY_SECONDS") or "").strip()
        if delay_raw:
            try:
                delay = float(delay_raw)
            except ValueError:
                delay = 0.0
            if delay > 0:
                if stop_event.wait(delay):
                    return
        while not stop_event.is_set():
            processed = self._drain_if_open(stop_event)
            if processed == 0:
                stop_event.wait(self._poll_interval_seconds)


def _is_retryable(policy: RetryPolicy, exc: Exception) -> bool:
    if isinstance(exc, IdempotencyLockConflictError):
        return policy.retry_on_lock_conflict
    if isinstance(exc, RetryableHandlerError):
        return policy.retry_on_handler_exception
    return False
