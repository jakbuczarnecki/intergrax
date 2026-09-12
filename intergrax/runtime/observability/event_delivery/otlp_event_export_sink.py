# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP transport adapter seam (W5-D) — no OpenTelemetry SDK in this layer."""

from __future__ import annotations

from intergrax.contracts.observability_export import OtlpTransportPort
from intergrax.runtime.events.runtime_event import RuntimeEvent


class OtlpEventExportSink:
    """
    ``EventExportSinkPort`` adapter toward an injected ``OtlpTransportPort``.

    Composition supplies transport when OTLP export is enabled.
    """

    def __init__(self, *, transport: OtlpTransportPort | None = None) -> None:
        self._transport = transport
        self._closed = False

    async def export(self, event: RuntimeEvent) -> None:
        if self._closed or self._transport is None:
            return
        self._transport.export(event)

    async def flush(self) -> None:
        if self._closed or self._transport is None:
            return
        self._transport.flush()

    async def close(self) -> None:
        if self._closed:
            return
        if self._transport is not None:
            self._transport.close()
        self._closed = True
