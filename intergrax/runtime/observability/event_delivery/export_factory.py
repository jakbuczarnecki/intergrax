# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default export sink factory (W5-D) — plain object, not a singleton."""

from __future__ import annotations

from intergrax.contracts.event_delivery import EventExportSinkPort
from intergrax.contracts.observability_export import (
    ExporterKind,
    ObservabilityExportProfile,
    OtlpTransportPort,
)
from intergrax.runtime.observability.event_delivery.noop_event_export_sink import (
    NoopEventExportSink,
)
from intergrax.runtime.observability.event_delivery.otlp_event_export_sink import (
    OtlpEventExportSink,
)
from intergrax.runtime.observability.event_delivery.recording_event_export_sink import (
    RecordingEventExportSink,
)


class ObservabilityExportSinkFactory:
    """Maps frozen export profile to a new ``EventExportSinkPort`` per call."""

    def __init__(self, *, otlp_transport: OtlpTransportPort | None = None) -> None:
        self._otlp_transport = otlp_transport

    def create(
        self,
        profile: ObservabilityExportProfile,
    ) -> EventExportSinkPort:
        if not profile.enabled or profile.exporter_kind is ExporterKind.NOOP:
            return NoopEventExportSink()
        if profile.exporter_kind is ExporterKind.RECORDING:
            return RecordingEventExportSink()
        if profile.exporter_kind in (ExporterKind.OTLP, ExporterKind.DISTRIBUTED_OTLP):
            return OtlpEventExportSink(transport=self._otlp_transport)
        return NoopEventExportSink()
