# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Observe provider termination and map to physical CAS (W4-D)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
    ExternalOperationState,
)
from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    ExternalOperationTerminationPort,
    TerminationOutcome,
    TerminationResult,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    finalize_operation_intent,
    mark_physical_terminal,
    request_operation_cancellation,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)


def cancellation_physical_state_from_termination(
    result: TerminationResult,
) -> ExternalOperationPhysicalState:
    if result.outcome in (
        TerminationOutcome.PHYSICAL_STOP_CONFIRMED,
        TerminationOutcome.TRANSPORT_CLOSED,
    ):
        return ExternalOperationPhysicalState.CANCELLED
    return ExternalOperationPhysicalState.UNKNOWN


def _run_awaitable_sync(awaitable: Awaitable[TerminationResult]) -> TerminationResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


async def dispatch_provider_termination(
    identity: ExternalOperationIdentity,
    *,
    termination_port: ExternalOperationTerminationPort | None,
    capabilities: ExternalOperationCapabilities,
) -> TerminationResult:
    if termination_port is None:
        return TerminationResult.not_supported()
    if (
        not capabilities.supports_native_cancel
        and not capabilities.supports_stream_abort
        and not capabilities.supports_remote_termination
    ):
        return TerminationResult.not_supported()
    try:
        return await termination_port.terminate(identity)
    except BaseException:
        return TerminationResult.signal_failed()


def dispatch_provider_termination_sync(
    identity: ExternalOperationIdentity,
    *,
    termination_port: ExternalOperationTerminationPort | None,
    capabilities: ExternalOperationCapabilities,
) -> TerminationResult:
    return _run_awaitable_sync(
        dispatch_provider_termination(
            identity,
            termination_port=termination_port,
            capabilities=capabilities,
        ),
    )


def mark_observed_cancellation_terminal(
    store: ExternalOperationStateStore,
    *,
    operation_id: str,
    expected_revision: int,
    termination_result: TerminationResult,
    finalize_intent: bool = True,
) -> ExternalOperationState:
    physical = cancellation_physical_state_from_termination(termination_result)
    intent: ExternalOperationIntentState | None = None
    if physical is ExternalOperationPhysicalState.UNKNOWN:
        intent = ExternalOperationIntentState.TERMINATING
    record = mark_physical_terminal(
        store,
        operation_id=operation_id,
        expected_revision=expected_revision,
        physical_state=physical,
        intent_state=intent,
    )
    if finalize_intent and physical is ExternalOperationPhysicalState.CANCELLED:
        return finalize_operation_intent(
            store,
            operation_id=operation_id,
            expected_revision=record.revision,
        )
    return record


def request_cancel_dispatch_terminate_and_observe(
    store: ExternalOperationStateStore,
    identity: ExternalOperationIdentity,
    *,
    expected_revision: int,
    cancellation_port: object | None,
    termination_port: ExternalOperationTerminationPort | None,
    capabilities: ExternalOperationCapabilities,
) -> ExternalOperationState:
    """Intent CAS → optional cancel port → termination → physical CAS."""
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationCancellationPort,
    )

    port: ExternalOperationCancellationPort | None = None
    if cancellation_port is not None:
        if not isinstance(cancellation_port, ExternalOperationCancellationPort):
            raise TypeError(
                "cancellation_port must implement ExternalOperationCancellationPort",
            )
        port = cancellation_port
    intent_record = request_operation_cancellation(
        store,
        operation_id=identity.operation_id,
        cancellation_port=port,
    )
    revision = intent_record.revision if expected_revision < 0 else expected_revision
    termination = dispatch_provider_termination_sync(
        identity,
        termination_port=termination_port,
        capabilities=capabilities,
    )
    return mark_observed_cancellation_terminal(
        store,
        operation_id=identity.operation_id,
        expected_revision=revision,
        termination_result=termination,
    )


def timeout_physical_state(*, exceeded_deadline: bool) -> ExternalOperationPhysicalState:
    """Timeout is never cancellation — FAILED or UNKNOWN only."""
    if exceeded_deadline:
        return ExternalOperationPhysicalState.UNKNOWN
    return ExternalOperationPhysicalState.FAILED


def register_stream_transport_closer(
    registry: dict[str, Callable[[], None]],
    *,
    operation_id: str,
    closer: Callable[[], None],
) -> None:
    if type(operation_id) is not str or not operation_id:
        raise ValueError("operation_id must be a non-empty str")
    registry[operation_id] = closer


def pop_stream_transport_closer(
    registry: dict[str, Callable[[], None]],
    *,
    operation_id: str,
) -> Callable[[], None] | None:
    return registry.pop(operation_id, None)
