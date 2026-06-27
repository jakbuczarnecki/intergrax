# © Artur Czarnecki. All rights reserved.

"""Typed LKW domain evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

_LKW_INDEX_SUMMARY_V1 = "lkw.index_summary.v1"
_LKW_SEARCH_SUMMARY_V1 = "lkw.search_summary.v1"
_LKW_SYNTHESIZE_SUMMARY_V1 = "lkw.synthesize_summary.v1"


def _query_digest(query: str | None) -> str | None:
    text = str(query or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class LkwIndexSummaryDiagnostic(DiagnosticPayload):
    accepted_count: int
    rejected_count: int
    ingested_count: int
    chunk_count: int
    source_count: int
    rejected_reasons: tuple[str, ...] = ()
    raw_tool_reason: str | None = None

    @classmethod
    def schema_id(cls) -> str:
        return _LKW_INDEX_SUMMARY_V1

    def redact(self) -> LkwIndexSummaryDiagnostic:
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


@dataclass(frozen=True)
class LkwSearchSummaryDiagnostic(DiagnosticPayload):
    num_results: int
    evidence_count: int
    query_digest: str | None = None
    query_text: str | None = None
    raw_tool_reason: str | None = None
    source_refs: tuple[str, ...] = ()
    tenant_id: str | None = None
    workspace_id: str | None = None

    @classmethod
    def schema_id(cls) -> str:
        return _LKW_SEARCH_SUMMARY_V1

    def redact(self) -> LkwSearchSummaryDiagnostic:
        return LkwSearchSummaryDiagnostic(
            num_results=self.num_results,
            evidence_count=self.evidence_count,
            query_digest=self.query_digest,
            query_text=None,
            raw_tool_reason=self.raw_tool_reason,
            source_refs=self.source_refs,
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_results": self.num_results,
            "evidence_count": self.evidence_count,
        }
        if self.query_digest:
            payload["query_digest"] = self.query_digest
        if self.query_text:
            payload["query_text"] = self.query_text
        if self.raw_tool_reason:
            payload["raw_tool_reason"] = self.raw_tool_reason
        if self.source_refs:
            payload["source_refs"] = list(self.source_refs)
        if self.tenant_id:
            payload["tenant_id"] = self.tenant_id
        if self.workspace_id:
            payload["workspace_id"] = self.workspace_id
        return payload


@dataclass(frozen=True)
class LkwSynthesizeSummaryDiagnostic(DiagnosticPayload):
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
        return _LKW_SYNTHESIZE_SUMMARY_V1

    def redact(self) -> LkwSynthesizeSummaryDiagnostic:
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


def index_diagnostic_from_output(output: dict[str, object]) -> LkwIndexSummaryDiagnostic:
    summary = _as_dict(output.get("ingest_summary"))
    accepted = summary.get("accepted_paths") or []
    rejected = summary.get("rejected_paths") or []
    ingested = summary.get("ingested") or []
    accepted_count = len(accepted) if isinstance(accepted, list) else 0
    rejected_count = len(rejected) if isinstance(rejected, list) else 0
    ingested_count = sum(
        1 for item in ingested if isinstance(item, dict) and item.get("status") == "success"
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
    return LkwIndexSummaryDiagnostic(
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        ingested_count=ingested_count,
        chunk_count=chunk_count,
        source_count=source_count,
        rejected_reasons=tuple(rejected_reasons),
        raw_tool_reason=raw_tool_reason,
    )


def search_diagnostic_from_output(output: dict[str, object]) -> LkwSearchSummaryDiagnostic:
    summary = _as_dict(output.get("search_summary"))
    evidence = summary.get("evidence") or []
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    num_results = int(summary.get("num_results") or 0)
    query = summary.get("query")
    query_text = str(query).strip() if query is not None and str(query).strip() else None
    source_refs: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            source_path = item.get("source_path") or item.get("source")
            if source_path and str(source_path) not in source_refs:
                source_refs.append(str(source_path))
    collection_id = summary.get("collection_id")
    workspace_id = str(collection_id).strip() if collection_id else None
    raw_tool_reason = summary.get("raw_tool_reason")
    return LkwSearchSummaryDiagnostic(
        num_results=num_results,
        evidence_count=evidence_count,
        query_digest=_query_digest(query_text),
        query_text=query_text,
        raw_tool_reason=str(raw_tool_reason) if raw_tool_reason else None,
        source_refs=tuple(source_refs),
        workspace_id=workspace_id,
    )


def synthesize_diagnostic_from_output(output: dict[str, object]) -> LkwSynthesizeSummaryDiagnostic:
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
    return LkwSynthesizeSummaryDiagnostic(
        write_status=write_status,
        shadow_write=shadow_write,
        source_evidence_count=num_evidence_items,
        artifact_path=artifact_path,
        artifact_ref=artifact_ref,
        reason=reason,
        content_missing=content_missing if reason == "content_missing" else None,
    )
