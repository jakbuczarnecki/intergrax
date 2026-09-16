# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference in-memory delivery admission provider (ME-10-R1)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryAdmissionResult,
    CapabilityHandoffDeliveryAdmissionVerdict,
    CapabilityHandoffEnvelope,
    CapabilityHandoffIdentityConflictError,
)


@dataclass
class InMemoryCapabilityHandoffDeliveryAdmission:
    """Reference provider: idempotent duplicate admission within process scope only."""

    _by_handoff_id: dict[str, CapabilityHandoffEnvelope] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def admit(self, envelope: CapabilityHandoffEnvelope) -> CapabilityHandoffDeliveryAdmissionResult:
        with self._lock:
            existing = self._by_handoff_id.get(envelope.handoff_id)
            if existing is None:
                self._by_handoff_id[envelope.handoff_id] = envelope
                return CapabilityHandoffDeliveryAdmissionResult(
                    verdict=CapabilityHandoffDeliveryAdmissionVerdict.ADMITTED_NEW,
                    handoff_id=envelope.handoff_id,
                )
            if existing == envelope:
                return CapabilityHandoffDeliveryAdmissionResult(
                    verdict=CapabilityHandoffDeliveryAdmissionVerdict.DUPLICATE_IDENTICAL_PAYLOAD,
                    handoff_id=envelope.handoff_id,
                )
            raise CapabilityHandoffIdentityConflictError(
                "handoff_id already admitted with a different envelope payload",
            )

    def admitted_handoff_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._by_handoff_id))


__all__ = ["InMemoryCapabilityHandoffDeliveryAdmission"]
