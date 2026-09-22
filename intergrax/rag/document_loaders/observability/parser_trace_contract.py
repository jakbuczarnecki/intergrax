# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed document parser pipeline trace (RAG observability boundary)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.structured_json_value import StructuredJsonObject, StructuredJsonValue


class ParserAttemptStatus(StrEnum):
    SKIPPED_UNAVAILABLE = "skipped_unavailable"
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ParserTraceAttempt:
    parser_id: str
    status: ParserAttemptStatus
    latency_ms: float | None = None
    num_documents: int | None = None
    error: str | None = None

    def to_structured_object(self) -> StructuredJsonObject:
        payload: StructuredJsonObject = {
            "parser_id": self.parser_id,
            "status": self.status.value,
        }
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.num_documents is not None:
            payload["num_documents"] = self.num_documents
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class DocumentParserTrace:
    parser_id: str | None
    attempts: tuple[ParserTraceAttempt, ...]
    latency_ms: float | None = None

    def to_logging_extra_value(self) -> StructuredJsonObject:
        """JSON-safe payload for ``logging.Logger`` extra (not a second canonical model)."""
        payload: StructuredJsonObject = {
            "attempts": [attempt.to_structured_object() for attempt in self.attempts],
        }
        if self.parser_id is not None:
            payload["parser_id"] = self.parser_id
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        return payload


def _optional_float(value: StructuredJsonValue) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return None


def _optional_int(value: StructuredJsonValue) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def document_parser_trace_from_structured_value(
    value: StructuredJsonValue,
) -> DocumentParserTrace | None:
    if not isinstance(value, Mapping):
        return None
    attempts_raw = value.get("attempts")
    if not isinstance(attempts_raw, list):
        return None
    attempts: list[ParserTraceAttempt] = []
    for item in attempts_raw:
        if not isinstance(item, Mapping):
            return None
        parser_id_raw = item.get("parser_id")
        status_raw = item.get("status")
        if not isinstance(parser_id_raw, str) or not isinstance(status_raw, str):
            return None
        try:
            status = ParserAttemptStatus(status_raw)
        except ValueError:
            return None
        attempts.append(
            ParserTraceAttempt(
                parser_id=parser_id_raw,
                status=status,
                latency_ms=_optional_float(item.get("latency_ms")),
                num_documents=_optional_int(item.get("num_documents")),
                error=str(item["error"]) if isinstance(item.get("error"), str) else None,
            )
        )
    parser_id = value.get("parser_id")
    resolved_parser_id = parser_id if isinstance(parser_id, str) else None
    return DocumentParserTrace(
        parser_id=resolved_parser_id,
        attempts=tuple(attempts),
        latency_ms=_optional_float(value.get("latency_ms")),
    )


def parser_trace_from_tags(tags: Mapping[str, StructuredJsonValue]) -> DocumentParserTrace | None:
    embedded = tags.get("integration_parser_trace")
    if embedded is None:
        return None
    return document_parser_trace_from_structured_value(embedded)
