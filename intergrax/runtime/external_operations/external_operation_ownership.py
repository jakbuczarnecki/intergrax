# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External operation ownership helpers (W4-C) — not a central manager."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationCancellationPort,
    ExternalOperationIntentState,
    ExternalOperationNotFoundError,
    ExternalOperationPhysicalState,
    ExternalOperationState,
)
from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
    ExternalOperationStateUpdate,
    StaleExternalOperationStateError,
)


class ExternalOperationStartSuppressedError(RuntimeError):
    """External call must not start (cancellation won the race before submit)."""


class ExternalOperationRetryBlockedError(RuntimeError):
    """Physical retry blocked until cancellation ownership is decided."""


@dataclass(frozen=True, slots=True)
class ProcessLocalExternalOperationOwner:
    """Opaque owner token for one runtime process."""

    token: str

    @staticmethod
    def mint() -> ProcessLocalExternalOperationOwner:
        return ProcessLocalExternalOperationOwner(token=f"owner_{uuid.uuid4().hex}")


def request_operation_cancellation(
    store: ExternalOperationStateStore,
    *,
    operation_id: str,
    cancellation_port: ExternalOperationCancellationPort | None = None,
) -> ExternalOperationState:
    """Idempotent cancellation request on durable intent state."""
    record = store.load(operation_id)
    if record is None:
        record = store.create_if_absent(operation_id)
    if record.intent_state in (
        ExternalOperationIntentState.CANCELLATION_REQUESTED,
        ExternalOperationIntentState.TERMINATING,
        ExternalOperationIntentState.TERMINATED,
    ):
        return record
    updated = store.compare_and_set(
        operation_id,
        expected_revision=record.revision,
        update=ExternalOperationStateUpdate(
            intent_state=ExternalOperationIntentState.CANCELLATION_REQUESTED,
        ),
    )
    if cancellation_port is not None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(cancellation_port.request_cancel(operation_id))
        else:
            loop.create_task(cancellation_port.request_cancel(operation_id))
    return updated


def assert_may_begin_physical_attempt(
    store: ExternalOperationStateStore,
    identity: ExternalOperationIdentity,
) -> ExternalOperationState:
    """Fail closed when cancellation precedes physical submit (case 1)."""
    operation_id = identity.operation_id
    record = store.load(operation_id)
    if record is None:
        return store.create_if_absent(operation_id)
    if record.intent_state is not ExternalOperationIntentState.ACTIVE:
        raise ExternalOperationStartSuppressedError(
            f"external operation {operation_id!r} not active: {record.intent_state.value}",
        )
    if identity.physical_attempt_sequence > 1:
        if record.intent_state is not ExternalOperationIntentState.ACTIVE:
            raise ExternalOperationRetryBlockedError(
                f"retry blocked for {operation_id!r} under intent {record.intent_state.value}",
            )
        if record.physical_state not in (
            ExternalOperationPhysicalState.FAILED,
            ExternalOperationPhysicalState.CANCELLED,
            ExternalOperationPhysicalState.UNKNOWN,
        ):
            raise ExternalOperationRetryBlockedError(
                f"retry blocked for {operation_id!r}: physical {record.physical_state.value}",
            )
    return record


def mark_physical_running(
    store: ExternalOperationStateStore,
    *,
    operation_id: str,
    expected_revision: int,
    owner: ProcessLocalExternalOperationOwner,
) -> ExternalOperationState:
    return store.compare_and_set(
        operation_id,
        expected_revision=expected_revision,
        update=ExternalOperationStateUpdate(
            physical_state=ExternalOperationPhysicalState.RUNNING,
            owner_token=owner.token,
        ),
    )


def mark_physical_terminal(
    store: ExternalOperationStateStore,
    *,
    operation_id: str,
    expected_revision: int,
    physical_state: ExternalOperationPhysicalState,
    intent_state: ExternalOperationIntentState | None = None,
) -> ExternalOperationState:
    if physical_state not in (
        ExternalOperationPhysicalState.SUCCEEDED,
        ExternalOperationPhysicalState.FAILED,
        ExternalOperationPhysicalState.CANCELLED,
        ExternalOperationPhysicalState.UNKNOWN,
    ):
        raise ValueError("physical_state must be terminal")
    try:
        return store.compare_and_set(
            operation_id,
            expected_revision=expected_revision,
            update=ExternalOperationStateUpdate(
                intent_state=intent_state,
                physical_state=physical_state,
            ),
        )
    except StaleExternalOperationStateError as stale:
        current = store.load(operation_id)
        if current is None:
            raise stale
        if (
            current.physical_state in (
                ExternalOperationPhysicalState.SUCCEEDED,
                ExternalOperationPhysicalState.FAILED,
                ExternalOperationPhysicalState.CANCELLED,
                ExternalOperationPhysicalState.UNKNOWN,
            )
            and current.physical_state is physical_state
        ):
            return current
        raise stale


def finalize_operation_intent(
    store: ExternalOperationStateStore,
    *,
    operation_id: str,
    expected_revision: int,
) -> ExternalOperationState:
    return store.compare_and_set(
        operation_id,
        expected_revision=expected_revision,
        update=ExternalOperationStateUpdate(
            intent_state=ExternalOperationIntentState.TERMINATED,
            clear_owner_token=True,
        ),
    )


def reconcile_orphaned_running_operations(
    store: ExternalOperationStateStore,
    *,
    active_owner: ProcessLocalExternalOperationOwner,
) -> tuple[ExternalOperationState, ...]:
    """Mark RUNNING records without the active owner as UNKNOWN (worker loss)."""
    reconciled: list[ExternalOperationState] = []
    for record in store.list_running():
        if record.owner_token == active_owner.token:
            continue
        try:
            updated = store.compare_and_set(
                record.operation_id,
                expected_revision=record.revision,
                update=ExternalOperationStateUpdate(
                    physical_state=ExternalOperationPhysicalState.UNKNOWN,
                    intent_state=ExternalOperationIntentState.TERMINATING,
                ),
            )
        except (StaleExternalOperationStateError, ExternalOperationNotFoundError):
            continue
        reconciled.append(updated)
    return tuple(reconciled)
