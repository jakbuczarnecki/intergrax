# © Artur Czarnecki. All rights reserved.

"""Serialize typed DiagnosticPayload instances into AgentStepRecord.diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


def serialize_diagnostic_payload(
    payload: DiagnosticPayload,
    *,
    redact: bool = True,
) -> dict[str, Any]:
    """Return JSON-safe payload body keyed for diagnostics storage."""
    effective = payload.redact() if redact else payload
    return effective.to_dict()


def merge_diagnostic_payloads(
    diagnostics: dict[str, Any] | None,
    payloads: Sequence[DiagnosticPayload],
    *,
    redact: bool = True,
) -> dict[str, Any]:
    """Merge typed payloads into diagnostics keyed by stable schema_id."""
    merged = dict(diagnostics or {})
    for payload in payloads:
        merged[payload.schema_id()] = serialize_diagnostic_payload(payload, redact=redact)
    return merged
