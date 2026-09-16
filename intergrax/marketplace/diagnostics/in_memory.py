# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from threading import Lock

from intergrax.contracts.marketplace.diagnostics import MarketplaceDiagnosticEvent


class InMemoryMarketplaceDiagnosticObserver:
    """Reference/test sink — not a production telemetry authority."""

    def __init__(self, observer_id: str = "test.marketplace.diagnostics.in_memory") -> None:
        self._observer_id = observer_id
        self._events: list[MarketplaceDiagnosticEvent] = []
        self._lock = Lock()

    @property
    def observer_id(self) -> str:
        return self._observer_id

    @property
    def events(self) -> tuple[MarketplaceDiagnosticEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def emit(self, event: MarketplaceDiagnosticEvent) -> None:
        with self._lock:
            self._events.append(event)


__all__ = ["InMemoryMarketplaceDiagnosticObserver"]
