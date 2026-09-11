# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP SDK adapter seam (W5-C) — no full OTLP client in this layer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from intergrax.runtime.events.runtime_event import RuntimeEvent

OtlpExportDelegate = Callable[[RuntimeEvent], Awaitable[None]]


class OtlpEventExportSink:
    """
    ``EventExportSinkPort`` adapter toward an injected OTLP SDK delegate.

    Composition supplies the delegate when OTLP export is enabled.
    """

    def __init__(self, *, export_delegate: OtlpExportDelegate | None = None) -> None:
        self._export_delegate = export_delegate
        self._closed = False

    async def export(self, event: RuntimeEvent) -> None:
        if self._closed or self._export_delegate is None:
            return
        await self._export_delegate(event)

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        self._closed = True
