# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import random
from collections.abc import Callable

from celery import Celery

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.lease_claim import LeaseOwnership, StaleClaimError
from intergrax.distributed.contracts.rate_limiter import (
    DistributedRateLimiter,
    RateLimitResult,
)
from intergrax.queueing.contracts.task_queue import TaskHandle, is_terminal_task_status
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
    DocumentStoreTaskQueue,
    DocumentStoreWakeClaimNotGrantedError,
    TaskDeliveryMismatchError,
    TaskRowClaimContext,
)
from intergrax.queueing.providers.document_store.task_row_lease import invocation_claim_from_row
from intergrax.queueing.worker.execution import (
    IdempotencyLockConflictError,
    RetryableHandlerError,
    execute_logical_task,
)
from intergrax.queueing.worker.rate_limit_event import RateLimitEvent
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.result_codec import encode_logical_task_result
from intergrax.queueing.worker.retry_backoff import calculate_retry_countdown
from intergrax.queueing.worker.retry_event import RetryEvent
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
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


def _durable_wake_terminal_ack() -> bytes:
    from pydantic import BaseModel

    from intergrax.tools.execution_models import ToolExecutionResult

    class _TerminalAck(BaseModel):
        terminal_duplicate: bool = True

    return encode_logical_task_result(ToolExecutionResult.ok(_TerminalAck()))


def _execute_document_store_wake(
    *,
    self,
    document_store_task_queue: DocumentStoreTaskQueue,
    document_task_id: str,
    logical_task_name: str,
    tenant_id: str,
    run_id: str,
    payload: bytes,
    idempotency_key: str | None,
    registry: TaskExecutionRegistry,
    idempotency_store: IdempotencyStore | None,
    lock_ttl_seconds: int | None,
    completed_ttl_seconds: int | None,
    identity_persistence: BackgroundExecutionIdentityPersistence,
    causal_evidence_persistence: CausalEvidencePersistence,
    attempt_lifecycle: AttemptLifecycleService,
    execution_terminal: ExecutionTerminalService,
) -> bytes:
    handle = TaskHandle(
        task_id=document_task_id,
        provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
        tenant_id=tenant_id,
    )
    status = document_store_task_queue.get_status_if_present(handle)
    if status is not None and is_terminal_task_status(status):
        return _durable_wake_terminal_ack()

    lease_seconds = lock_ttl_seconds or document_store_task_queue._default_lease_seconds
    owner_id = f"celery-wake-{self.request.id}"
    wake_claim = document_store_task_queue.claim_task_for_wake(
        handle,
        claim_context=TaskRowClaimContext(owner_id=owner_id, lease_seconds=lease_seconds),
    )
    if wake_claim is None:
        status = document_store_task_queue.get_status_if_present(handle)
        if status is not None and is_terminal_task_status(status):
            return _durable_wake_terminal_ack()
        raise DocumentStoreWakeClaimNotGrantedError(
            f"document store wake could not claim task {document_task_id!r}",
        )

    presented: LeaseOwnership = wake_claim.presented_claim
    wake_request = wake_claim.request
    try:
        document_store_task_queue.assert_delivery_matches(
            handle,
            tenant_id=tenant_id,
            identity=wake_request.identity,
        )
    except TaskDeliveryMismatchError as exc:
        raise DocumentStoreWakeClaimNotGrantedError(str(exc)) from exc

    transport_ref = BackgroundTransportExecutionRef(
        tenant_id=tenant_id,
        provider="celery",
        transport_task_id=str(self.request.id),
    )
    reentry = admit_background_execution_reentry(
        transport_ref=transport_ref,
        identity_persistence=identity_persistence,
        attempt_lifecycle=attempt_lifecycle,
        execution_terminal=execution_terminal,
    )
    if reentry.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED:
        try:
            document_store_task_queue.mark_succeeded(
                handle,
                output=None,
                presented_claim=presented,
            )
        except StaleClaimError as exc:
            raise DocumentStoreWakeClaimNotGrantedError(str(exc)) from exc
        return _durable_wake_terminal_ack()

    execution_identity = reentry.identity
    row = document_store_task_queue._load(handle)
    row_claim = invocation_claim_from_row(row, tenant_id=tenant_id) if row is not None else None

    try:
        result = admit_background_execution_handler(
            transport_ref=transport_ref,
            execution_identity=execution_identity,
            causal_evidence_persistence=causal_evidence_persistence,
            handler=lambda: execute_logical_task(
                registry=registry,
                logical_task_name=logical_task_name,
                tenant_id=execution_identity.tenant_id,
                run_id=str(execution_identity.run_id),
                payload=payload,
                idempotency_key=idempotency_key,
                idempotency_store=idempotency_store,
                lease_seconds=lock_ttl_seconds,
                completed_ttl_seconds=completed_ttl_seconds,
                execution_identity=execution_identity,
                existing_claim=row_claim,
            ),
        )
    except Exception as exc:
        try:
            document_store_task_queue.mark_failed(
                handle,
                error_message=str(exc),
                presented_claim=presented,
            )
        except StaleClaimError as stale:
            raise DocumentStoreWakeClaimNotGrantedError(str(stale)) from stale
        raise

    try:
        if result.success:
            output = None
            if result.output is not None:
                output = result.output.model_dump_json().encode("utf-8")
            document_store_task_queue.mark_succeeded(
                handle,
                output=output,
                presented_claim=presented,
            )
        else:
            message = result.error.error_message if result.error is not None else "handler_failed"
            document_store_task_queue.mark_failed(
                handle,
                error_message=message,
                presented_claim=presented,
            )
    except StaleClaimError as exc:
        raise DocumentStoreWakeClaimNotGrantedError(str(exc)) from exc

    return encode_logical_task_result(result)


def register_dispatcher_task(
    app: Celery,
    registry: TaskExecutionRegistry,
    idempotency_store: IdempotencyStore | None = None,
    rate_limiter: DistributedRateLimiter | None = None,
    *,
    lock_ttl_seconds: int | None = None,
    completed_ttl_seconds: int | None = None,
    retry_policy: RetryPolicy | None = None,
    on_retry_scheduled: Callable[[RetryEvent], None] | None = None,
    rate_limit_config: Callable[[str], tuple[int, float]] | None = None,
    on_rate_limited: Callable[[RateLimitEvent], None] | None = None,
    identity_persistence: BackgroundExecutionIdentityPersistence,
    causal_evidence_persistence: CausalEvidencePersistence,
    attempt_lifecycle: AttemptLifecycleService,
    execution_terminal: ExecutionTerminalService,
    document_store_task_queue: DocumentStoreTaskQueue | None = None,
) -> None:
    """
    Registers the generic execution task into Celery app.

    This must be called during worker composition phase.
    """

    if retry_policy is not None:
        retry_policy.validate()

    if rate_limiter is not None and retry_policy is None:
        raise ValueError("retry_policy must be provided when rate_limiter is enabled.")

    resolved_identity_persistence = identity_persistence
    resolved_causal_evidence_persistence = causal_evidence_persistence

    @app.task(name="intergrax.execute", bind=True)
    def intergrax_execute(
        self,
        logical_task_name: str,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None = None,
        document_task_id: str | None = None,
    ) -> bytes:
        """
        Generic execution entrypoint.

        Dispatches execution to registered logical task handler.
        """

        # -------------------------
        # Tier-0: Distributed Rate Limiting
        # -------------------------
        if rate_limiter is not None:
            if rate_limit_config is None:
                raise ValueError("rate_limit_config must be provided when rate_limiter is enabled.")

            capacity, refill_rate_per_second = rate_limit_config(logical_task_name)

            result: RateLimitResult = rate_limiter.acquire(
                tenant_id=tenant_id,
                key=logical_task_name,
                capacity=int(capacity),
                refill_rate_per_second=float(refill_rate_per_second),
            )

            if not result.allowed:
                current_retries: int = self.request.retries

                if current_retries >= retry_policy.max_retries:
                    raise RuntimeError("rate_limited: max_retries reached")

                base_countdown: float = float(result.retry_after_seconds)

                # ---- RateLimit hook (attached to task, not closure) ----
                rate_limit_hook: Callable[[RateLimitEvent], None] | None = self.on_rate_limited  # type: ignore[attr-defined]

                if rate_limit_hook is not None:
                    rate_limit_event = RateLimitEvent(
                        logical_task_name=logical_task_name,
                        tenant_id=tenant_id,
                        retry_after_seconds=base_countdown,
                        remaining_tokens=result.remaining_tokens,
                        current_retries=current_retries,
                        max_retries=retry_policy.max_retries,
                    )
                    rate_limit_hook(rate_limit_event)

                # ---- Jitter (20%) ----
                jitter_window: float = base_countdown * 0.2
                jitter: float = random.uniform(0.0, jitter_window)
                countdown: float = base_countdown + jitter

                # ---- Retry hook ----
                retry_hook: Callable[[RetryEvent], None] | None = self.on_retry_scheduled  # type: ignore[attr-defined]

                if retry_hook is not None:
                    retry_event = RetryEvent(
                        logical_task_name=logical_task_name,
                        exception_type="RateLimited",
                        current_retries=current_retries,
                        max_retries=retry_policy.max_retries,
                        countdown_seconds=countdown,
                        reason="rate_limited",
                    )
                    retry_hook(retry_event)

                raise self.retry(
                    exc=RuntimeError("rate_limited"),
                    countdown=countdown,
                )

        # -------------------------
        # Core Execution
        # -------------------------
        try:
            if document_task_id is not None and document_store_task_queue is not None:
                return _execute_document_store_wake(
                    self=self,
                    document_store_task_queue=document_store_task_queue,
                    document_task_id=document_task_id,
                    logical_task_name=logical_task_name,
                    tenant_id=tenant_id,
                    run_id=run_id,
                    payload=payload,
                    idempotency_key=idempotency_key,
                    registry=registry,
                    idempotency_store=idempotency_store or document_store_task_queue._idempotency_store,
                    lock_ttl_seconds=lock_ttl_seconds,
                    completed_ttl_seconds=completed_ttl_seconds,
                    identity_persistence=resolved_identity_persistence,
                    causal_evidence_persistence=resolved_causal_evidence_persistence,
                    attempt_lifecycle=attempt_lifecycle,
                    execution_terminal=execution_terminal,
                )
            transport_ref = BackgroundTransportExecutionRef(
                tenant_id=tenant_id,
                provider="celery",
                transport_task_id=str(self.request.id),
            )
            reentry = admit_background_execution_reentry(
                transport_ref=transport_ref,
                identity_persistence=resolved_identity_persistence,
                attempt_lifecycle=attempt_lifecycle,
                execution_terminal=execution_terminal,
            )
            if reentry.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED:
                from pydantic import BaseModel

                from intergrax.tools.execution_models import ToolExecutionResult

                class _TerminalAck(BaseModel):
                    terminal_duplicate: bool = True

                return encode_logical_task_result(
                    ToolExecutionResult.ok(_TerminalAck()),
                )
            execution_identity = reentry.identity
            return encode_logical_task_result(
                admit_background_execution_handler(
                    transport_ref=transport_ref,
                    execution_identity=execution_identity,
                    causal_evidence_persistence=resolved_causal_evidence_persistence,
                    handler=lambda: execute_logical_task(
                        registry=registry,
                        logical_task_name=logical_task_name,
                        tenant_id=execution_identity.tenant_id,
                        run_id=str(execution_identity.run_id),
                        payload=payload,
                        idempotency_key=idempotency_key,
                        idempotency_store=idempotency_store,
                        lease_seconds=lock_ttl_seconds,
                        completed_ttl_seconds=completed_ttl_seconds,
                        execution_identity=execution_identity,
                    ),
                )
            )

        except (IdempotencyLockConflictError, RetryableHandlerError) as exc:
            if retry_policy is None:
                raise

            if isinstance(exc, IdempotencyLockConflictError):
                should_retry = retry_policy.retry_on_lock_conflict
                reason = "lock_conflict"
            else:
                should_retry = retry_policy.retry_on_handler_exception
                reason = "handler_transient"

            if not should_retry:
                raise

            current_retries: int = self.request.retries

            if current_retries >= retry_policy.max_retries:
                raise

            countdown: float = calculate_retry_countdown(
                policy=retry_policy,
                current_retries=current_retries,
            )

            retry_hook: Callable[[RetryEvent], None] | None = self.on_retry_scheduled  # type: ignore[attr-defined]

            if retry_hook is not None:
                retry_event = RetryEvent(
                    logical_task_name=logical_task_name,
                    exception_type=type(exc).__name__,
                    current_retries=current_retries,
                    max_retries=retry_policy.max_retries,
                    countdown_seconds=countdown,
                    reason=reason,
                )
                retry_hook(retry_event)

            raise self.retry(exc=exc, countdown=countdown)

    # ---------------------------------------------
    # Attach hooks to task instance (production-safe)
    # ---------------------------------------------
    intergrax_execute.on_rate_limited = on_rate_limited  # type: ignore[attr-defined]
    intergrax_execute.on_retry_scheduled = on_retry_scheduled  # type: ignore[attr-defined]
