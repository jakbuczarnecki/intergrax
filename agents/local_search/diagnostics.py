# © Artur Czarnecki. All rights reserved.

"""Typed search evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

_SEARCH_SUMMARY_V1 = "lkw.search_summary.v1"


class SearchSummaryReason(str, Enum):
    QUERY_MISSING = "query_missing"
    TOOL_GATEWAY_NOT_AVAILABLE = "tool_gateway_not_available"
    RETRIEVE_FAILED = "retrieve_failed"
    RETRIEVE_COMPLETE = "retrieve_complete"


def parse_search_summary_reason(
    value: object,
) -> SearchSummaryReason | None:
    if isinstance(value, SearchSummaryReason):
        return value

    if not isinstance(value, str):
        return None

    normalized = value.strip()
    if not normalized:
        return None

    try:
        return SearchSummaryReason(normalized)
    except ValueError:
        return None


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _query_digest(query: str | None) -> str | None:
    text = str(query or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SearchSummaryDiagnostic(DiagnosticPayload):
    num_results: int
    evidence_count: int
    query_digest: str | None = None
    query_text: str | None = None
    raw_tool_reason: str | None = None
    source_refs: tuple[str, ...] = ()
    tenant_id: str | None = None
    workspace_id: str | None = None
    used: bool | None = None
    reason: SearchSummaryReason | None = None

    @classmethod
    def schema_id(cls) -> str:
        return _SEARCH_SUMMARY_V1

    def redact(self) -> SearchSummaryDiagnostic:
        return SearchSummaryDiagnostic(
            num_results=self.num_results,
            evidence_count=self.evidence_count,
            query_digest=self.query_digest,
            query_text=None,
            raw_tool_reason=self.raw_tool_reason,
            source_refs=self.source_refs,
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            used=self.used,
            reason=self.reason,
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
        if self.used is not None:
            payload["used"] = self.used
        if self.reason is not None:
            payload["reason"] = self.reason.value
        return payload


def search_diagnostic_from_output(output: dict[str, object]) -> SearchSummaryDiagnostic:
    summary = _as_dict(output.get("search_summary"))
    evidence = summary.get("evidence") or []
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    num_results = int(summary.get("num_results") or 0)
    query = summary.get("query")
    query_text = (
        str(query).strip() if query is not None and str(query).strip() else None
    )
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
    used_raw = summary.get("used")
    used = used_raw if isinstance(used_raw, bool) else None
    reason = parse_search_summary_reason(summary.get("reason"))
    return SearchSummaryDiagnostic(
        num_results=num_results,
        evidence_count=evidence_count,
        query_digest=_query_digest(query_text),
        query_text=query_text,
        raw_tool_reason=str(raw_tool_reason) if raw_tool_reason else None,
        source_refs=tuple(source_refs),
        workspace_id=workspace_id,
        used=used,
        reason=reason,
    )
