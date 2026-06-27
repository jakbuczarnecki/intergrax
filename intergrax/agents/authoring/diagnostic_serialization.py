# © Artur Czarnecki. All rights reserved.

"""Serialize typed DiagnosticPayload instances into AgentStepRecord.diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from intergrax.contracts.agent_run_trace import AgentRunTrace
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


def aggregate_step_diagnostics(trace: AgentRunTrace) -> dict[str, dict[str, Any]]:
    """Merge step diagnostics keyed by schema_id (last step wins per schema)."""
    aggregated: dict[str, dict[str, Any]] = {}
    for step in trace.steps:
        for schema_id, payload in (step.diagnostics or {}).items():
            if isinstance(payload, dict):
                aggregated[schema_id] = dict(payload)
    return aggregated
