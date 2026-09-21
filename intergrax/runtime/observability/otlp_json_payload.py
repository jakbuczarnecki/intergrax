# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded OTLP JSON wire shapes for observability export (not a full OTLP SDK)."""

from __future__ import annotations

from typing import TypedDict


class OtlpArrayValue(TypedDict):
    values: list[OtlpAnyValue]


class OtlpAnyValue(TypedDict, total=False):
    """Subset of OTLP JSON ``AnyValue`` used by Intergrax exporters."""

    stringValue: str
    boolValue: bool
    intValue: str
    doubleValue: float
    arrayValue: OtlpArrayValue


class OtlpKeyValue(TypedDict):
    key: str
    value: OtlpAnyValue


class OtlpLogBody(TypedDict):
    stringValue: str


class OtlpLogRecord(TypedDict):
    timeUnixNano: str
    severityText: str
    body: OtlpLogBody
    attributes: list[OtlpKeyValue]


class OtlpScope(TypedDict):
    name: str


class OtlpScopeLogs(TypedDict):
    scope: OtlpScope
    logRecords: list[OtlpLogRecord]


class OtlpResource(TypedDict):
    attributes: list[OtlpKeyValue]


class OtlpResourceLogs(TypedDict):
    resource: OtlpResource
    scopeLogs: list[OtlpScopeLogs]


class OtlpLogsJsonPayload(TypedDict):
    """Top-level OTLP JSON document for log export."""

    resourceLogs: list[OtlpResourceLogs]


__all__ = [
    "OtlpAnyValue",
    "OtlpArrayValue",
    "OtlpKeyValue",
    "OtlpLogBody",
    "OtlpLogRecord",
    "OtlpLogsJsonPayload",
    "OtlpResource",
    "OtlpResourceLogs",
    "OtlpScope",
    "OtlpScopeLogs",
]
