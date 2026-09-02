# © Artur Czarnecki. All rights reserved.

"""Curated LKW evidence read model derived from AgentRunTrace step diagnostics."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_trace import AgentRunTrace

LKW_EVIDENCE_SCHEMA_VERSION = "lkw_evidence.v1"

LKW_TYPED_DIAGNOSTIC_SCHEMA_IDS: frozenset[str] = frozenset(
    {
        "lkw.index_summary.v1",
        "lkw.search_summary.v1",
        "lkw.synthesize_summary.v1",
        "lkw.web_search_summary.v1",
    }
)

_UNSAFE_DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {
        "query_text",
        "text",
        "content",
        "raw_chunks",
        "chunks",
        "document",
        "documents",
    }
)


class LkwEvidenceSliceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lkw_evidence.v1"] = LKW_EVIDENCE_SCHEMA_VERSION
    capability: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    terminal_status: str | None = None
    diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _sanitize_lkw_diagnostic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in _UNSAFE_DIAGNOSTIC_KEYS
    }


def collect_lkw_diagnostics_from_trace(trace: AgentRunTrace) -> dict[str, dict[str, Any]]:
    """Extract typed LKW diagnostics from Plane B step records (last step wins per schema)."""
    collected: dict[str, dict[str, Any]] = {}
    for step in trace.steps:
        for schema_id, payload in (step.diagnostics or {}).items():
            if schema_id not in LKW_TYPED_DIAGNOSTIC_SCHEMA_IDS:
                continue
            if not isinstance(payload, dict):
                continue
            collected[schema_id] = _sanitize_lkw_diagnostic_payload(payload)
    return collected


def collect_lkw_diagnostics_from_step_diagnostics(
    step_diagnostics: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Extract typed LKW diagnostics from aggregated step diagnostics mapping."""
    collected: dict[str, dict[str, Any]] = {}
    for schema_id, payload in (step_diagnostics or {}).items():
        if schema_id not in LKW_TYPED_DIAGNOSTIC_SCHEMA_IDS:
            continue
        if not isinstance(payload, dict):
            continue
        collected[schema_id] = _sanitize_lkw_diagnostic_payload(payload)
    return collected


def build_lkw_evidence_slice(
    trace: AgentRunTrace,
    *,
    capability: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    terminal_status: str | None = None,
) -> LkwEvidenceSliceV1:
    """Build curated evidence slice from an AgentRunTrace."""
    return LkwEvidenceSliceV1(
        capability=capability,
        agent_id=agent_id,
        run_id=run_id or trace.run_id or None,
        task_id=task_id,
        terminal_status=terminal_status,
        diagnostics=collect_lkw_diagnostics_from_trace(trace),
    )


def build_lkw_evidence_slice_from_step_diagnostics(
    step_diagnostics: dict[str, Any] | None,
    *,
    capability: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    terminal_status: str | None = None,
) -> LkwEvidenceSliceV1:
    """Build curated evidence slice from aggregated Plane B step diagnostics."""
    return LkwEvidenceSliceV1(
        capability=capability,
        agent_id=agent_id,
        run_id=run_id,
        task_id=task_id,
        terminal_status=terminal_status,
        diagnostics=collect_lkw_diagnostics_from_step_diagnostics(step_diagnostics),
    )
