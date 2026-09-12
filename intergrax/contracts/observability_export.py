# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Observability export composition contracts (W5-D)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.contracts.event_delivery import EventExportSinkPort
    from intergrax.runtime.events.runtime_event import RuntimeEvent


class ExporterKind(StrEnum):
    NOOP = "noop"
    OTLP = "otlp"
    RECORDING = "recording"


@dataclass(frozen=True, slots=True)
class ObservabilityExportProfile:
    enabled: bool
    exporter_kind: ExporterKind


class ExportError(Exception):
    """Export transport failure — must not propagate to execution plane."""


@runtime_checkable
class EventExportSinkFactoryPort(Protocol):
    """Profile → export sink only (no registry, no global lifecycle)."""

    def create(
        self,
        profile: ObservabilityExportProfile,
    ) -> EventExportSinkPort: ...


@runtime_checkable
class OtlpTransportPort(Protocol):
    """Pluggable OTLP (or vendor) transport — sync seam for adapter injection."""

    def export(
        self,
        event: RuntimeEvent,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...
