# © Artur Czarnecki. All rights reserved.

"""Live runtime signal bridge (ECP-PROD.1)."""

from __future__ import annotations

from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


class CapacityEventBridge:
    """Subscribe to runtime events and feed the capacity signal collector."""

    def __init__(
        self,
        collector: CapacitySignalCollector,
        event_bus: RuntimeEventBus,
    ) -> None:
        self._collector = collector
        self._event_bus = event_bus
        self._subscription_id: str | None = None

    def attach(self) -> None:
        if self._subscription_id is not None:
            return

        def _on_backpressure(_event: RuntimeEvent) -> None:
            self._collector.record_backpressure()

        self._subscription_id = self._event_bus.subscribe(
            _on_backpressure,
            event_types={RuntimeEventType.GRAPH_BACKPRESSURE},
        )

    def detach(self) -> None:
        if self._subscription_id is None:
            return
        self._event_bus.unsubscribe(self._subscription_id)
        self._subscription_id = None
