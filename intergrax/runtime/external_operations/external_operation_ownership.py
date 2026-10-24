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


class ExternalOperationFencedCommitRejectedError(RuntimeError):
    """Superseded worker cannot publish or mutate durable external-operation state."""


class ExternalOperationCertifiedOwnerLossRequiredError(RuntimeError):
    """RUNNING→UNKNOWN recovery requires canonical certified owner-loss evidence (parent L3)."""


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
    tenant_id: str,
    operation_id: str,
    cancellation_port: ExternalOperationCancellationPort | None = None,
) -> ExternalOperationState:
    """Idempotent cancellation request on durable intent state."""
    record = store.load(tenant_id, operation_id)
    if record is None:
        record = store.create_if_absent(tenant_id, operation_id)
    if record.intent_state in (
        ExternalOperationIntentState.CANCELLATION_REQUESTED,
        ExternalOperationIntentState.TERMINATING,
        ExternalOperationIntentState.TERMINATED,
    ):
        return record
    updated = store.compare_and_set(
        tenant_id,
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
    tenant_id = identity.tenant_id
    operation_id = identity.operation_id
    record = store.load(tenant_id, operation_id)
    if record is None:
        return store.create_if_absent(tenant_id, operation_id)
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
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    owner: ProcessLocalExternalOperationOwner,
) -> ExternalOperationState:
    return store.compare_and_set(
        tenant_id,
        operation_id,
        expected_revision=expected_revision,
        update=ExternalOperationStateUpdate(
            physical_state=ExternalOperationPhysicalState.RUNNING,
            owner_token=owner.token,
        ),
    )


_TERMINAL_PHYSICAL = frozenset(
    {
        ExternalOperationPhysicalState.SUCCEEDED,
        ExternalOperationPhysicalState.FAILED,
        ExternalOperationPhysicalState.CANCELLED,
        ExternalOperationPhysicalState.UNKNOWN,
    },
)


def mark_physical_terminal_if_owner_matches(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    physical_state: ExternalOperationPhysicalState,
    owner: ProcessLocalExternalOperationOwner,
    intent_state: ExternalOperationIntentState | None = None,
) -> ExternalOperationState:
    """CAS terminal commit only when ``owner`` still holds the RUNNING record."""
    if physical_state not in _TERMINAL_PHYSICAL:
        raise ValueError("physical_state must be terminal")
    current = store.load(tenant_id, operation_id)
    if current is None:
        raise ExternalOperationNotFoundError(operation_id)
    if current.physical_state in _TERMINAL_PHYSICAL:
        if current.physical_state is physical_state:
            if (
                current.owner_token is not None
                and current.owner_token != owner.token
            ):
                raise ExternalOperationFencedCommitRejectedError(
                    f"stale owner cannot observe terminal for {operation_id!r}",
                )
            return current
        raise ExternalOperationFencedCommitRejectedError(
            f"late terminal rejected for {operation_id!r}: "
            f"existing {current.physical_state.value}",
        )
    if (
        current.physical_state is ExternalOperationPhysicalState.RUNNING
        and current.owner_token is not None
        and current.owner_token != owner.token
    ):
        raise ExternalOperationFencedCommitRejectedError(
            f"stale owner cannot commit terminal for {operation_id!r}",
        )
    return mark_physical_terminal(
        store,
        tenant_id=tenant_id,
        operation_id=operation_id,
        expected_revision=expected_revision,
        physical_state=physical_state,
        intent_state=intent_state,
    )


def mark_physical_terminal(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
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
            tenant_id,
            operation_id,
            expected_revision=expected_revision,
            update=ExternalOperationStateUpdate(
                intent_state=intent_state,
                physical_state=physical_state,
            ),
        )
    except StaleExternalOperationStateError as stale:
        current = store.load(tenant_id, operation_id)
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
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
) -> ExternalOperationState:
    return store.compare_and_set(
        tenant_id,
        operation_id,
        expected_revision=expected_revision,
        update=ExternalOperationStateUpdate(
            intent_state=ExternalOperationIntentState.TERMINATED,
            clear_owner_token=True,
        ),
    )


def finalize_operation_intent_if_owner_matches(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    owner: ProcessLocalExternalOperationOwner,
) -> ExternalOperationState:
    """Clear owner and terminate intent only for the durable record holder."""
    current = store.load(tenant_id, operation_id)
    if current is None:
        raise ExternalOperationNotFoundError(operation_id)
    if (
        current.owner_token is not None
        and current.owner_token != owner.token
    ):
        raise ExternalOperationFencedCommitRejectedError(
            f"stale owner cannot finalize intent for {operation_id!r}",
        )
    return finalize_operation_intent(
        store,
        tenant_id=tenant_id,
        operation_id=operation_id,
        expected_revision=expected_revision,
    )


def recover_running_operation_after_certified_owner_loss(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    evidence_id: str | None = None,
) -> ExternalOperationState:
    """Apply staged certified owner-loss evidence (RUNNING→UNKNOWN) when configured."""
    current = store.load(tenant_id, operation_id)
    if current is None:
        raise ExternalOperationNotFoundError(operation_id)
    if current.physical_state is not ExternalOperationPhysicalState.RUNNING:
        return current
    if evidence_id is None:
        raise ExternalOperationCertifiedOwnerLossRequiredError(
            f"certified owner-loss evidence required for RUNNING→UNKNOWN on {operation_id!r}",
        )
    return store.apply_certified_owner_loss(
        tenant_id,
        operation_id,
        expected_revision=expected_revision,
        evidence_id=evidence_id,
    )


def reconcile_orphaned_running_operations(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    active_owner: ProcessLocalExternalOperationOwner,
) -> tuple[ExternalOperationState, ...]:
    """No-op: process-local owner token mismatch is not proof of owner death (TOOL-LIFE-X Batch C)."""
    del store, tenant_id, active_owner
    return ()
