# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenTelemetry SDK transport adapter for ``OtlpTransportPort`` (W5-E)."""

from __future__ import annotations

from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GrpcOTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as HttpOTLPLogExporter,
)
from opentelemetry.sdk._logs import LogData, LogRecord, LoggerProvider
from opentelemetry.sdk._logs.export import LogExportResult, SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

from intergrax.contracts.observability_export import (
    OtlpExportConfiguration,
    OtlpProtocol,
    OtlpTransportError,
    OtlpTransportPort,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.observability.export_boundary import (
    ObservabilityExportEnvelope,
    envelope_from_runtime_event,
)
from intergrax.runtime.observability.exporters.otlp.otlp_configuration import (
    validate_otlp_export_configuration,
)

_EXPORT_SCOPE_NAME = "intergrax.runtime.event_delivery"
_DEFAULT_SERVICE_NAME = "intergrax"


def _envelope_attributes(envelope: ObservabilityExportEnvelope) -> dict[str, str | int | bool]:
    mapped: dict[str, str | int | bool] = {
        "intergrax.schema_version": envelope.schema_version,
        "intergrax.record_kind": envelope.record_kind.value,
        "intergrax.status": envelope.status.value,
    }
    for key, value in (
        ("intergrax.run_id", envelope.run_id),
        ("intergrax.task_id", envelope.task_id),
        ("intergrax.attempt_id", envelope.attempt_id),
        ("intergrax.execution_id", envelope.execution_id),
        ("intergrax.agent_id", envelope.agent_id),
        ("intergrax.capability", envelope.capability),
        ("intergrax.tool_id", envelope.tool_id),
        ("intergrax.event_type", envelope.event_type),
        ("intergrax.schema_id", envelope.schema_id),
        ("intergrax.source_schema_id", envelope.source_schema_id),
        ("intergrax.correlation_id", envelope.correlation_id),
        ("intergrax.event_id", envelope.event_id),
        ("intergrax.tenant_id", envelope.tenant_id),
        ("intergrax.workspace_id", envelope.workspace_id),
    ):
        if value:
            mapped[key] = value
    if envelope.latency_ms is not None:
        mapped["intergrax.latency_ms"] = envelope.latency_ms
    for count_key, count_value in sorted(envelope.counts.items()):
        mapped[f"intergrax.counts.{count_key}"] = count_value
    return mapped


def runtime_event_to_otlp_log_record(event: RuntimeEvent) -> LogRecord:
    """Map a runtime event to an OTLP SDK log record (test and diagnostics seam)."""
    return _log_record_from_envelope(envelope_from_runtime_event(event))


def _log_record_from_envelope(envelope: ObservabilityExportEnvelope) -> LogRecord:
    body = envelope.event_type or envelope.record_kind.value
    return LogRecord(
        timestamp=int(envelope.recorded_at.timestamp() * 1_000_000_000),
        trace_id=0,
        span_id=0,
        trace_flags=None,
        severity_text=envelope.status.value.upper(),
        severity_number=SeverityNumber.INFO,
        body=body,
        attributes=_envelope_attributes(envelope),
    )


class OtlpTransport(OtlpTransportPort):
    """
    Maps ``RuntimeEvent`` → OTLP log records → SDK exporter.

    No queue, retry, buffering, or circuit breaking — bounded delivery owns backpressure.
    """

    def __init__(
        self,
        config: OtlpExportConfiguration,
        *,
        service_name: str = _DEFAULT_SERVICE_NAME,
    ) -> None:
        validated = validate_otlp_export_configuration(config)
        self._config = validated
        self._closed = False
        self._resource = Resource.create(
            {"service.name": service_name.strip() or _DEFAULT_SERVICE_NAME},
        )
        self._scope = InstrumentationScope(_EXPORT_SCOPE_NAME, "")
        timeout_ms = max(1, int(validated.timeout_seconds * 1000))
        if validated.protocol is OtlpProtocol.GRPC:
            exporter = GrpcOTLPLogExporter(
                endpoint=validated.endpoint,
                timeout=timeout_ms,
            )
        else:
            exporter = HttpOTLPLogExporter(
                endpoint=validated.endpoint,
                timeout=timeout_ms,
            )
        provider = LoggerProvider(resource=self._resource)
        provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        self._provider = provider
        self._exporter = exporter

    def export(self, event: RuntimeEvent) -> None:
        if self._closed:
            return
        envelope = envelope_from_runtime_event(event)
        record = _log_record_from_envelope(envelope)
        try:
            result = self._exporter.export((LogData(record, self._scope),))
        except Exception as exc:
            raise OtlpTransportError(str(exc)) from exc
        if result is not LogExportResult.SUCCESS:
            raise OtlpTransportError("otlp log export rejected by exporter")

    def flush(self) -> None:
        if self._closed:
            return
        self._provider.force_flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._provider.shutdown()
        self._closed = True
