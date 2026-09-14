# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minimal in-process health state for bounded delivery subsystems."""

from __future__ import annotations

import threading

from intergrax.contracts.event_delivery import EventSinkHealthPort, EventSinkHealthState


class MutableEventSinkHealth:
    """Thread-safe health holder owned by ``BoundedEventSink``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = EventSinkHealthState.HEALTHY

    def health_state(self) -> EventSinkHealthState:
        with self._lock:
            return self._state

    def mark_unhealthy(self) -> None:
        with self._lock:
            self._state = EventSinkHealthState.UNHEALTHY


def assert_event_sink_health_port(health: EventSinkHealthPort) -> EventSinkHealthPort:
    return health
