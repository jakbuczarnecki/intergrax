# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local terminal sink for observability delivery (W5-B2 composition)."""

from __future__ import annotations

import threading

from intergrax.contracts.event_delivery import (
    DeliverableEvent,
    EventDeliveryDisposition,
    EventDeliveryResult,
    EventPriority,
    effective_event_delivery_obligation,
)
from intergrax.runtime.observability.event_delivery.enterprise_default_event_delivery_obligation_policy import (
    EnterpriseDefaultEventDeliveryObligationPolicy,
)

_DEFAULT_OBLIGATION_POLICY = EnterpriseDefaultEventDeliveryObligationPolicy()


class AcceptingObservabilityEventSink:
    """
    Synchronous terminal ``EventSinkPort`` for production composition wiring.

    Retains only an acceptance counter for diagnostics; durable evidence remains
    on ``RuntimeEventPersistence`` (orthogonal transport path).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def accepted_count(self) -> int:
        with self._lock:
            return self._accepted

    def publish(
        self,
        event: DeliverableEvent,
        *,
        priority: EventPriority,
        deadline: float | None = None,
    ) -> EventDeliveryResult:
        obligation = effective_event_delivery_obligation(priority, _DEFAULT_OBLIGATION_POLICY)
        with self._lock:
            if self._closed:
                return EventDeliveryResult(
                    disposition=EventDeliveryDisposition.REJECTED,
                    priority=priority,
                    buffered_depth=0,
                    obligation=obligation,
                )
            self._accepted += 1
            depth = self._accepted
        return EventDeliveryResult(
            disposition=EventDeliveryDisposition.ACCEPTED,
            priority=priority,
            buffered_depth=depth,
            obligation=obligation,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
