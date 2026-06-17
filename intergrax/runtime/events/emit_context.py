# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Emit context for HOS public APIs (OBS-EVOL-9.3 · SAR-01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus


@dataclass(frozen=True, slots=True)
class EmitContext:
    """Correlation bundle passed to ``emit_domain_signal`` / ``emit_platform_event``."""

    task_id: str
    run_id: str
    tenant_id: str | None = None
    correlation_id: str = ""
    parent_event_id: str | None = None
    traceparent: str | None = None
    tracestate: str | None = None
    bus: RuntimeEventBus | None = None
    production_mode: bool = False

    @property
    def effective_correlation_id(self) -> str:
        return self.correlation_id or self.task_id
