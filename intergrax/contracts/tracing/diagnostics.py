# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plane B diagnostic payload contract (Harness Observability Spine)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.tracing.values import TraceObject, validate_trace_value

DEFAULT_REDACTED_TEXT: str = "[REDACTED]"


class DiagnosticPayload(ABC):
    """
    Typed diagnostic payload contract (production).

    Rules:
    - schema_id: stable identifier (never reused for different semantics)
    - schema_version: bump only when schema changes
    - to_dict(): MUST return a JSON-safe object (see TraceObject / TraceValue)
    """

    @classmethod
    @abstractmethod
    def schema_id(cls) -> str:
        raise NotImplementedError

    @classmethod
    def schema_version(cls) -> int:
        return 1

    @abstractmethod
    def to_dict(self) -> TraceObject:
        raise NotImplementedError

    @abstractmethod
    def redact(self) -> DiagnosticPayload:
        raise NotImplementedError

    def serialized_dict(self) -> TraceObject:
        """Validated JSON-safe export for evidence and audit sinks."""
        payload = self.to_dict()
        validated = validate_trace_value(payload, field_name="payload")
        if not isinstance(validated, dict):
            raise ValueError("payload must serialize to a JSON object")
        return validated
