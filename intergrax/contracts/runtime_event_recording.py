# © Artur Czarnecki. All rights reserved.

"""Synchronous runtime event recording port (observability contract surface)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.runtime_event import RuntimeEvent


@runtime_checkable
class RuntimeEventRecorderPort(Protocol):
    """Minimal non-authoritative recording seam for canonical ``RuntimeEvent`` evidence."""

    def record(
        self,
        event: RuntimeEvent,
        *,
        tenant_id: str | None = None,
    ) -> None: ...


__all__ = ["RuntimeEventRecorderPort"]
