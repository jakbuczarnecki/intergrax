# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker wake-up signal and admission result contracts (AW-4A)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from intergrax.contracts.autonomous_work._validation import require_aware_utc
from intergrax.contracts.autonomous_work.continuity import WorkContinuityState
from intergrax.contracts.autonomous_work.ids import (
    WakeUpId,
    WorkerInstanceId,
    validate_wake_up_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.references import (
    WakeUpCorrelationRef,
    WakeUpSourceRef,
    validate_wake_up_correlation_ref,
    validate_wake_up_source_ref,
)
from intergrax.contracts.autonomous_work.worker import WorkerInstance

CANONICAL_WORKER_WAKE_UP_SOURCE_KINDS: Final = (
    "EXTERNAL_EVENT",
    "QUEUE_DELIVERY",
    "SCHEDULE",
    "HUMAN_CONTINUATION",
    "DEPENDENCY_RECOVERY",
    "OPERATOR",
    "RECOVERY_TIMER",
)


class WorkerWakeUpSourceKind(StrEnum):
    """Semantic wake-up trigger class — transport-agnostic."""

    EXTERNAL_EVENT = "EXTERNAL_EVENT"
    QUEUE_DELIVERY = "QUEUE_DELIVERY"
    SCHEDULE = "SCHEDULE"
    HUMAN_CONTINUATION = "HUMAN_CONTINUATION"
    DEPENDENCY_RECOVERY = "DEPENDENCY_RECOVERY"
    OPERATOR = "OPERATOR"
    RECOVERY_TIMER = "RECOVERY_TIMER"


class WorkerWakeUpDisposition(StrEnum):
    """Typed admission outcome for one wake-up delivery attempt."""

    ACCEPTED = "ACCEPTED"
    CONFLICT = "CONFLICT"
    DUPLICATE = "DUPLICATE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class WorkerWakeUpSignal:
    """Provider-neutral wake-up trigger normalized by transport adapters."""

    wake_up_id: WakeUpId
    worker_instance_id: WorkerInstanceId
    source_kind: WorkerWakeUpSourceKind
    source_ref: WakeUpSourceRef
    occurred_at: datetime
    delivery_identity: WakeUpId
    correlation_ref: WakeUpCorrelationRef | None = None

    def __post_init__(self) -> None:
        validate_wake_up_id(self.wake_up_id)
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.source_kind) is not WorkerWakeUpSourceKind:
            raise TypeError("source_kind must be WorkerWakeUpSourceKind")
        validate_wake_up_source_ref(self.source_ref)
        object.__setattr__(
            self,
            "occurred_at",
            require_aware_utc(self.occurred_at, label="occurred_at"),
        )
        validate_wake_up_id(self.delivery_identity)


@dataclass(frozen=True, slots=True)
class WorkerWakeUpReceipt:
    """Durable idempotency record for one accepted wake-up delivery."""

    worker_instance_id: WorkerInstanceId
    wake_up_id: WakeUpId
    source_kind: WorkerWakeUpSourceKind
    source_ref: WakeUpSourceRef
    occurred_at: datetime
    accepted_at: datetime
    delivery_identity: WakeUpId
    correlation_ref: WakeUpCorrelationRef | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        validate_wake_up_id(self.wake_up_id)
        if type(self.source_kind) is not WorkerWakeUpSourceKind:
            raise TypeError("source_kind must be WorkerWakeUpSourceKind")
        validate_wake_up_source_ref(self.source_ref)
        object.__setattr__(
            self,
            "occurred_at",
            require_aware_utc(self.occurred_at, label="occurred_at"),
        )
        object.__setattr__(
            self,
            "accepted_at",
            require_aware_utc(self.accepted_at, label="accepted_at"),
        )
        validate_wake_up_id(self.delivery_identity)
        if self.correlation_ref is not None:
            validate_wake_up_correlation_ref(self.correlation_ref)


@dataclass(frozen=True, slots=True)
class WorkerWakeUpContext:
    """Orientation restored after wake-up admission."""

    worker_instance: WorkerInstance
    wake_up_signal: WorkerWakeUpSignal
    continuity_state: WorkContinuityState | None
    accepted_at: datetime
    disposition: WorkerWakeUpDisposition
    receipt: WorkerWakeUpReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.worker_instance) is not WorkerInstance:
            raise TypeError("worker_instance must be WorkerInstance")
        if type(self.wake_up_signal) is not WorkerWakeUpSignal:
            raise TypeError("wake_up_signal must be WorkerWakeUpSignal")
        if self.continuity_state is not None:
            if type(self.continuity_state) is not WorkContinuityState:
                raise TypeError("continuity_state must be WorkContinuityState")
        object.__setattr__(
            self,
            "accepted_at",
            require_aware_utc(self.accepted_at, label="accepted_at"),
        )
        if type(self.disposition) is not WorkerWakeUpDisposition:
            raise TypeError("disposition must be WorkerWakeUpDisposition")
        if self.receipt is not None:
            if type(self.receipt) is not WorkerWakeUpReceipt:
                raise TypeError("receipt must be WorkerWakeUpReceipt")


@dataclass(frozen=True, slots=True)
class WakeUpLogicalSignalIdentity:
    """Canonical immutable identity for one logical wake-up signal.

    ``delivery_identity`` is excluded: it identifies a transport delivery attempt,
    not the logical wake-up. Idempotency is keyed by
    ``(worker_instance_id, wake_up_id)``. ``accepted_at`` is excluded because it
    belongs to admission, not signal identity.
    """

    worker_instance_id: WorkerInstanceId
    wake_up_id: WakeUpId
    source_kind: WorkerWakeUpSourceKind
    source_ref: WakeUpSourceRef
    occurred_at: datetime
    correlation_ref: WakeUpCorrelationRef | None = None


def wake_up_logical_identity_from_signal(
    signal: WorkerWakeUpSignal,
) -> WakeUpLogicalSignalIdentity:
    """Project the canonical logical identity from a wake-up signal."""
    return WakeUpLogicalSignalIdentity(
        worker_instance_id=signal.worker_instance_id,
        wake_up_id=signal.wake_up_id,
        source_kind=signal.source_kind,
        source_ref=signal.source_ref,
        occurred_at=signal.occurred_at,
        correlation_ref=signal.correlation_ref,
    )


def wake_up_logical_identity_from_receipt(
    receipt: WorkerWakeUpReceipt,
) -> WakeUpLogicalSignalIdentity:
    """Project the canonical logical identity from a durable wake-up receipt."""
    return WakeUpLogicalSignalIdentity(
        worker_instance_id=receipt.worker_instance_id,
        wake_up_id=receipt.wake_up_id,
        source_kind=receipt.source_kind,
        source_ref=receipt.source_ref,
        occurred_at=receipt.occurred_at,
        correlation_ref=receipt.correlation_ref,
    )


def wake_up_signals_logically_equivalent(
    left: WorkerWakeUpSignal | WorkerWakeUpReceipt,
    right: WorkerWakeUpSignal | WorkerWakeUpReceipt,
) -> bool:
    """Return whether two signal projections represent the same logical wake-up."""
    left_identity = (
        wake_up_logical_identity_from_signal(left)
        if isinstance(left, WorkerWakeUpSignal)
        else wake_up_logical_identity_from_receipt(left)
    )
    right_identity = (
        wake_up_logical_identity_from_signal(right)
        if isinstance(right, WorkerWakeUpSignal)
        else wake_up_logical_identity_from_receipt(right)
    )
    return left_identity == right_identity


def worker_wake_up_signal_from_receipt(
    receipt: WorkerWakeUpReceipt,
) -> WorkerWakeUpSignal:
    """Reconstruct the canonical wake-up signal representation from a receipt."""
    return WorkerWakeUpSignal(
        wake_up_id=receipt.wake_up_id,
        worker_instance_id=receipt.worker_instance_id,
        source_kind=receipt.source_kind,
        source_ref=receipt.source_ref,
        occurred_at=receipt.occurred_at,
        delivery_identity=receipt.delivery_identity,
        correlation_ref=receipt.correlation_ref,
    )


@dataclass(frozen=True, slots=True)
class WorkerWakeUpResult:
    """Typed wake-up admission outcome."""

    disposition: WorkerWakeUpDisposition
    context: WorkerWakeUpContext | None

    def __post_init__(self) -> None:
        if type(self.disposition) is not WorkerWakeUpDisposition:
            raise TypeError("disposition must be WorkerWakeUpDisposition")
        if self.context is not None:
            if type(self.context) is not WorkerWakeUpContext:
                raise TypeError("context must be WorkerWakeUpContext")
