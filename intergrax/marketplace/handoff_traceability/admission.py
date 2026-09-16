# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference in-memory delivery lifecycle admission provider (ME-10-R2)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryAdmissionResult,
    CapabilityHandoffDeliveryAdmissionVerdict,
    CapabilityHandoffDeliveryLifecycleState,
    CapabilityHandoffDeliveryLifecycleTransitionError,
    CapabilityHandoffEnvelope,
    CapabilityHandoffIdentityConflictError,
)


@dataclass(frozen=True, slots=True)
class _InMemoryLifecycleEntry:
    envelope: CapabilityHandoffEnvelope
    state: CapabilityHandoffDeliveryLifecycleState


def _require_in_progress_for_transition(
    handoff_id: str,
    entry: _InMemoryLifecycleEntry | None,
    *,
    operation: str,
) -> _InMemoryLifecycleEntry:
    if entry is None:
        raise CapabilityHandoffDeliveryLifecycleTransitionError(
            f"handoff_id={handoff_id!r} current_state=absent attempted={operation}",
        )
    if entry.state is not CapabilityHandoffDeliveryLifecycleState.IN_PROGRESS:
        raise CapabilityHandoffDeliveryLifecycleTransitionError(
            f"handoff_id={handoff_id!r} current_state={entry.state.value} attempted={operation}",
        )
    return entry


@dataclass
class InMemoryCapabilityHandoffDeliveryAdmission:
    """Reference provider: process-local lifecycle semantics only — not distributed exactly-once."""

    _by_handoff_id: dict[str, _InMemoryLifecycleEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        with self._lock:
            existing = self._by_handoff_id.get(envelope.handoff_id)
            if existing is None:
                self._by_handoff_id[envelope.handoff_id] = _InMemoryLifecycleEntry(
                    envelope=envelope,
                    state=CapabilityHandoffDeliveryLifecycleState.IN_PROGRESS,
                )
                return CapabilityHandoffDeliveryAdmissionResult(
                    verdict=CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW,
                    handoff_id=envelope.handoff_id,
                )
            if existing.envelope != envelope:
                raise CapabilityHandoffIdentityConflictError(
                    "handoff_id already reserved or delivered with a different envelope payload",
                )
            if existing.state is CapabilityHandoffDeliveryLifecycleState.DELIVERED:
                return CapabilityHandoffDeliveryAdmissionResult(
                    verdict=CapabilityHandoffDeliveryAdmissionVerdict.ALREADY_DELIVERED_IDENTICAL,
                    handoff_id=envelope.handoff_id,
                )
            if existing.state is CapabilityHandoffDeliveryLifecycleState.IN_PROGRESS:
                return CapabilityHandoffDeliveryAdmissionResult(
                    verdict=CapabilityHandoffDeliveryAdmissionVerdict.IN_PROGRESS_IDENTICAL,
                    handoff_id=envelope.handoff_id,
                )
            self._by_handoff_id[envelope.handoff_id] = _InMemoryLifecycleEntry(
                envelope=envelope,
                state=CapabilityHandoffDeliveryLifecycleState.IN_PROGRESS,
            )
            return CapabilityHandoffDeliveryAdmissionResult(
                verdict=CapabilityHandoffDeliveryAdmissionVerdict.RESERVED_NEW,
                handoff_id=envelope.handoff_id,
            )

    def mark_delivered(self, handoff_id: str) -> None:
        with self._lock:
            entry = _require_in_progress_for_transition(
                handoff_id,
                self._by_handoff_id.get(handoff_id),
                operation="mark_delivered",
            )
            self._by_handoff_id[handoff_id] = _InMemoryLifecycleEntry(
                envelope=entry.envelope,
                state=CapabilityHandoffDeliveryLifecycleState.DELIVERED,
            )

    def mark_delivery_failed(self, handoff_id: str) -> None:
        with self._lock:
            entry = _require_in_progress_for_transition(
                handoff_id,
                self._by_handoff_id.get(handoff_id),
                operation="mark_delivery_failed",
            )
            self._by_handoff_id[handoff_id] = _InMemoryLifecycleEntry(
                envelope=entry.envelope,
                state=CapabilityHandoffDeliveryLifecycleState.FAILED_RETRYABLE,
            )

    def delivered_handoff_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(
                    handoff_id
                    for handoff_id, entry in self._by_handoff_id.items()
                    if entry.state is CapabilityHandoffDeliveryLifecycleState.DELIVERED
                ),
            )


__all__ = ["InMemoryCapabilityHandoffDeliveryAdmission"]
