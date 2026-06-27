# © Artur Czarnecki. All rights reserved.

"""Typed synthesize evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

_SYNTHESIZE_SUMMARY_V1 = "lkw.synthesize_summary.v1"


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class SynthesizeSummaryDiagnostic(DiagnosticPayload):
    write_status: str
    shadow_write: bool
    source_evidence_count: int
    artifact_path: str | None = None
    artifact_ref: str | None = None
    reason: str | None = None
    content_missing: bool | None = None
    raw_tool_reason: str | None = None

    @classmethod
    def schema_id(cls) -> str:
        return _SYNTHESIZE_SUMMARY_V1

    def redact(self) -> SynthesizeSummaryDiagnostic:
        return self

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "write_status": self.write_status,
            "shadow_write": self.shadow_write,
            "source_evidence_count": self.source_evidence_count,
        }
        if self.artifact_path:
            payload["artifact_path"] = self.artifact_path
        if self.artifact_ref:
            payload["artifact_ref"] = self.artifact_ref
        if self.reason:
            payload["reason"] = self.reason
        if self.content_missing is not None:
            payload["content_missing"] = self.content_missing
        if self.raw_tool_reason:
            payload["raw_tool_reason"] = self.raw_tool_reason
        return payload


def synthesize_diagnostic_from_output(output: dict[str, object]) -> SynthesizeSummaryDiagnostic:
    summary = _as_dict(output.get("synthesize_summary"))
    reason = str(summary.get("reason") or "unknown")
    used = bool(summary.get("used"))
    write_status = reason if reason else ("write_complete" if used else "write_failed")
    shadow_write = bool(summary.get("shadow_workspace"))
    num_evidence_items = int(summary.get("num_evidence_items") or 0)
    artifact_path_raw = summary.get("artifact_path")
    artifact_path = str(artifact_path_raw) if artifact_path_raw else None
    output_name = summary.get("output_name")
    artifact_ref = str(output_name).strip() if output_name else artifact_path
    content_missing = reason == "content_missing"
    return SynthesizeSummaryDiagnostic(
        write_status=write_status,
        shadow_write=shadow_write,
        source_evidence_count=num_evidence_items,
        artifact_path=artifact_path,
        artifact_ref=artifact_ref,
        reason=reason,
        content_missing=content_missing if reason == "content_missing" else None,
    )
