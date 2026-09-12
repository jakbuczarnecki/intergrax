# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""No-op export sink for feature-off, local dev, and production default handoff."""

from __future__ import annotations

from intergrax.runtime.events.runtime_event import RuntimeEvent


class NoopEventExportSink:
    """Accept, flush, and close without side effects."""

    async def export(self, event: RuntimeEvent) -> None:
        return None

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None
