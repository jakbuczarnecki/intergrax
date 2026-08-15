# © Artur Czarnecki. All rights reserved.

"""Typed index evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

_INDEX_SUMMARY_V1 = "lkw.index_summary.v1"


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class IndexSummaryDiagnostic(DiagnosticPayload):
    accepted_count: int
    rejected_count: int
    ingested_count: int
    chunk_count: int
    source_count: int
    rejected_reasons: tuple[str, ...] = ()
    raw_tool_reason: str | None = None

    @classmethod
    def schema_id(cls) -> str:
        return _INDEX_SUMMARY_V1

    def redact(self) -> IndexSummaryDiagnostic:
        return self

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "ingested_count": self.ingested_count,
            "chunk_count": self.chunk_count,
            "source_count": self.source_count,
        }
        if self.rejected_reasons:
            payload["rejected_reasons"] = list(self.rejected_reasons)
        if self.raw_tool_reason:
            payload["raw_tool_reason"] = self.raw_tool_reason
        return payload


def index_diagnostic_from_output(output: dict[str, object]) -> IndexSummaryDiagnostic:
    summary = _as_dict(output.get("ingest_summary"))
    accepted = summary.get("accepted_paths") or []
    rejected = summary.get("rejected_paths") or []
    ingested = summary.get("ingested") or []
    accepted_count = len(accepted) if isinstance(accepted, list) else 0
    rejected_count = len(rejected) if isinstance(rejected, list) else 0
    ingested_count = sum(
        1
        for item in ingested
        if isinstance(item, dict)
        and item.get("status") == "success"
        and item.get("used") is True
    )
    chunk_count = int(summary.get("num_chunks") or 0)
    source_count = accepted_count + rejected_count
    rejected_reasons: list[str] = []
    if isinstance(rejected, list):
        for item in rejected:
            if isinstance(item, dict):
                reason = item.get("reason")
                if reason and str(reason) not in rejected_reasons:
                    rejected_reasons.append(str(reason))
    raw_tool_reason: str | None = None
    if isinstance(ingested, list):
        for item in ingested:
            if isinstance(item, dict) and item.get("status") != "success":
                reason = item.get("reason")
                if reason:
                    raw_tool_reason = str(reason)
                    break
    return IndexSummaryDiagnostic(
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        ingested_count=ingested_count,
        chunk_count=chunk_count,
        source_count=source_count,
        rejected_reasons=tuple(rejected_reasons),
        raw_tool_reason=raw_tool_reason,
    )
