# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-process export recorder for W5-C qualification tests."""

from __future__ import annotations

import threading

from intergrax.runtime.events.runtime_event import RuntimeEvent


class RecordingEventExportSink:
    """Captures exported runtime events in enqueue order."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[RuntimeEvent] = []
        self._flush_count = 0
        self._closed = False

    @property
    def events(self) -> list[RuntimeEvent]:
        with self._lock:
            return list(self._events)

    @property
    def flush_count(self) -> int:
        with self._lock:
            return self._flush_count

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    async def export(self, event: RuntimeEvent) -> None:
        with self._lock:
            if self._closed:
                return
            self._events.append(event)

    async def flush(self) -> None:
        with self._lock:
            self._flush_count += 1

    async def close(self) -> None:
        with self._lock:
            self._closed = True
