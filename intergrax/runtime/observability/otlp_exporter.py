# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP observability export adapter (OBS-EXPORT-4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol, runtime_checkable

from intergrax.runtime.observability.export_attributes import (
    ObservabilityAttributeValue,
    SanitizedApplicationObservabilityAttributes,
)
from intergrax.runtime.observability.export_boundary import ObservabilityExportEnvelope

_EXPORT_SCOPE_NAME = "intergrax.observability.export"


@dataclass(frozen=True, slots=True)
class OtlpObservabilityExporterConfig:
    endpoint: str
    service_name: str
    service_version: str = ""
    environment: str = ""
    timeout_seconds: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class OtlpTransport(Protocol):
    async def send(
        self,
        payload: Mapping[str, Any],
        *,
        config: OtlpObservabilityExporterConfig,
    ) -> None: ...


def _timestamp_to_unix_nano(value: datetime) -> str:
    return str(int(value.timestamp() * 1_000_000_000))


def _otlp_attribute_value(value: ObservabilityAttributeValue) -> dict[str, Any]:
    if value is None:
        return {"stringValue": ""}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {
            "arrayValue": {
                "values": [{"stringValue": item} for item in value],
            }
        }
    return {"stringValue": str(value)}


def _string_attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _optional_string_attr(key: str, value: str) -> dict[str, Any] | None:
    if not value:
        return None
    return _string_attr(key, value)


def _map_sanitized_application_attributes(
    attributes: SanitizedApplicationObservabilityAttributes | None,
) -> list[dict[str, Any]]:
    if attributes is None:
        return []
    mapped: list[dict[str, Any]] = []
    for key, value in sorted(attributes.attributes.items()):
        mapped.append({"key": key, "value": _otlp_attribute_value(value)})
    if attributes.namespace:
        mapped.append(_string_attr("intergrax.application.namespace", attributes.namespace))
    return mapped


def _envelope_to_otlp_payload(
    envelope: ObservabilityExportEnvelope,
    *,
    config: OtlpObservabilityExporterConfig,
) -> dict[str, Any]:
    """Map a policy-sanitized export envelope to an OTLP-safe log record payload."""
    log_attributes: list[dict[str, Any]] = []

    for key, value in (
        ("intergrax.schema_version", envelope.schema_version),
        ("intergrax.record_kind", envelope.record_kind.value),
        ("intergrax.run_id", envelope.run_id),
        ("intergrax.task_id", envelope.task_id),
        ("intergrax.attempt_id", envelope.attempt_id),
        ("intergrax.execution_id", envelope.execution_id),
        ("intergrax.agent_id", envelope.agent_id),
        ("intergrax.capability", envelope.capability),
        ("intergrax.tool_id", envelope.tool_id),
        ("intergrax.event_type", envelope.event_type),
        ("intergrax.status", envelope.status.value),
        ("intergrax.schema_id", envelope.schema_id),
        ("intergrax.source_schema_id", envelope.source_schema_id),
        ("intergrax.correlation_id", envelope.correlation_id),
        ("intergrax.event_id", envelope.event_id),
        ("intergrax.artifact_ref", envelope.artifact_ref),
        ("intergrax.sha256", envelope.sha256),
        ("intergrax.safe_relative_path", envelope.safe_relative_path),
        ("intergrax.tenant_id", envelope.tenant_id),
        ("intergrax.workspace_id", envelope.workspace_id),
    ):
        attr = _optional_string_attr(key, value)
        if attr is not None:
            log_attributes.append(attr)

    if envelope.latency_ms is not None:
        log_attributes.append(
            {
                "key": "intergrax.latency_ms",
                "value": {"intValue": str(envelope.latency_ms)},
            }
        )

    for count_key, count_value in sorted(envelope.counts.items()):
        log_attributes.append(
            {
                "key": f"intergrax.counts.{count_key}",
                "value": {"intValue": str(count_value)},
            }
        )

    log_attributes.extend(_map_sanitized_application_attributes(envelope.sanitized_application_attributes))

    resource_attributes: list[dict[str, Any]] = [
        _string_attr("service.name", config.service_name),
    ]
    if config.service_version:
        resource_attributes.append(_string_attr("service.version", config.service_version))
    if config.environment:
        resource_attributes.append(_string_attr("deployment.environment", config.environment))

    log_record: dict[str, Any] = {
        "timeUnixNano": _timestamp_to_unix_nano(envelope.recorded_at),
        "severityText": envelope.status.value.upper(),
        "body": {"stringValue": envelope.event_type or envelope.record_kind.value},
        "attributes": log_attributes,
    }

    return {
        "resourceLogs": [
            {
                "resource": {"attributes": resource_attributes},
                "scopeLogs": [
                    {
                        "scope": {"name": _EXPORT_SCOPE_NAME},
                        "logRecords": [log_record],
                    }
                ],
            }
        ]
    }


class OtlpObservabilityExporter:
    """Remote OTLP export adapter for normalized observability export envelopes."""

    def __init__(
        self,
        config: OtlpObservabilityExporterConfig,
        transport: OtlpTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    @property
    def config(self) -> OtlpObservabilityExporterConfig:
        return self._config

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        payload = _envelope_to_otlp_payload(envelope, config=self._config)
        await self._transport.send(payload, config=self._config)
