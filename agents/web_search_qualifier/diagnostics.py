# © Artur Czarnecki. All rights reserved.

"""Typed web-search evidence payloads for AgentStepRecord.diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

_WEB_SEARCH_SUMMARY_V1 = "lkw.web_search_summary.v1"


@dataclass(frozen=True)
class WebSearchSummaryDiagnostic(DiagnosticPayload):
    used: bool
    reason: str
    provider_id: str | None = None
    actual_query: str | None = None
    provider_invoked_with_query: str | None = None
    selected_url: str | None = None
    selected_artifact_ref: str | None = None
    extracted_fact: str | None = None
    candidate_urls: tuple[str, ...] = ()
    search_status: str | None = None

    @classmethod
    def schema_id(cls) -> str:
        return _WEB_SEARCH_SUMMARY_V1

    def redact(self) -> WebSearchSummaryDiagnostic:
        return WebSearchSummaryDiagnostic(
            used=self.used,
            reason=self.reason,
            provider_id=self.provider_id,
            actual_query=self.actual_query,
            provider_invoked_with_query=self.provider_invoked_with_query,
            selected_url=self.selected_url,
            selected_artifact_ref=self.selected_artifact_ref,
            extracted_fact=self.extracted_fact,
            candidate_urls=self.candidate_urls,
            search_status=self.search_status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "reason": self.reason,
            "provider_id": self.provider_id,
            "actual_query": self.actual_query,
            "provider_invoked_with_query": self.provider_invoked_with_query,
            "selected_url": self.selected_url,
            "selected_artifact_ref": self.selected_artifact_ref,
            "extracted_fact": self.extracted_fact,
            "candidate_urls": list(self.candidate_urls),
            "search_status": self.search_status,
            "ops": "web_search_summary",
        }


def web_search_diagnostic_from_output(output: dict[str, object]) -> WebSearchSummaryDiagnostic:
    summary = output.get("web_search_summary")
    if not isinstance(summary, dict):
        return WebSearchSummaryDiagnostic(used=False, reason="summary_missing")
    raw_urls = summary.get("candidate_urls")
    urls: tuple[str, ...] = ()
    if isinstance(raw_urls, list):
        urls = tuple(str(item) for item in raw_urls if isinstance(item, str))
    return WebSearchSummaryDiagnostic(
        used=bool(summary.get("used")),
        reason=str(summary.get("reason") or "unknown"),
        provider_id=str(summary["provider_id"]) if isinstance(summary.get("provider_id"), str) else None,
        actual_query=str(summary["actual_query"]) if isinstance(summary.get("actual_query"), str) else None,
        provider_invoked_with_query=(
            str(summary["provider_invoked_with_query"])
            if isinstance(summary.get("provider_invoked_with_query"), str)
            else None
        ),
        selected_url=str(summary["selected_url"]) if isinstance(summary.get("selected_url"), str) else None,
        selected_artifact_ref=(
            str(summary["selected_artifact_ref"])
            if isinstance(summary.get("selected_artifact_ref"), str)
            else None
        ),
        extracted_fact=(
            str(summary["extracted_fact"]) if isinstance(summary.get("extracted_fact"), str) else None
        ),
        candidate_urls=urls,
        search_status=str(summary["search_status"]) if isinstance(summary.get("search_status"), str) else None,
    )


__all__ = ["WebSearchSummaryDiagnostic", "web_search_diagnostic_from_output"]
