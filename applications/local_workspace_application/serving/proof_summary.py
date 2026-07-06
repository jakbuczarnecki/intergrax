# © Artur Czarnecki. All rights reserved.

"""Redacted LKW pipeline proof summary derived from existing HTTP metadata (LKW-3C)."""

from __future__ import annotations

from typing import Any

LKW_PROOF_SUMMARY_KEY = "lkw_proof_summary.v1"
LKW_PIPELINE_CAPABILITY = "local.workspace.pipeline"
APP_RUN_SUMMARY_KEY = "application_run_summary.v1"
RUN_ARTIFACT_BUNDLE_KEY = "run_artifact_bundle.v1"
LKW_EVIDENCE_KEY = "lkw_evidence.v1"

_PIPELINE_AGENTS: tuple[str, ...] = (
    "local_indexer",
    "local_search",
    "local_synthesizer",
)

_POSITIVE_TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "completed"})

_SEARCH_SUMMARY_KEY = "lkw.search_summary.v1"
_SYNTHESIZE_SUMMARY_KEY = "lkw.synthesize_summary.v1"

_UNSAFE_TRACE_KEYS: frozenset[str] = frozenset({"full_trace", "agent_run_trace"})


def build_lkw_proof_summary(
    metadata: dict[str, Any],
    *,
    capability: str,
) -> dict[str, Any] | None:
    """Build redacted proof summary from curated metadata only."""
    if capability != LKW_PIPELINE_CAPABILITY:
        return None

    app_summary = _dict_metadata(metadata, APP_RUN_SUMMARY_KEY)
    evidence = _dict_metadata(metadata, LKW_EVIDENCE_KEY)
    bundle = _dict_metadata(metadata, RUN_ARTIFACT_BUNDLE_KEY)

    agent_order = _agent_order(app_summary)
    tool_calls_by_agent = _tool_calls_by_agent(app_summary)
    search_diag = _diagnostic(evidence, _SEARCH_SUMMARY_KEY)
    synth_diag = _diagnostic(evidence, _SYNTHESIZE_SUMMARY_KEY)

    evidence_block = _evidence_block(search_diag, synth_diag)
    synthesis_block = _synthesis_block(synth_diag)
    artifact_block = _artifact_block(bundle, synth_diag)

    summary: dict[str, Any] = {
        "schema_version": LKW_PROOF_SUMMARY_KEY,
        "capability": LKW_PIPELINE_CAPABILITY,
        "status": _proof_status(
            app_summary=app_summary,
            agent_order=agent_order,
            tool_calls_by_agent=tool_calls_by_agent,
            evidence=evidence_block,
            synthesis=synthesis_block,
            artifact=artifact_block,
        ),
        "agent_order": agent_order,
        "tool_calls_by_agent": tool_calls_by_agent,
        "evidence": evidence_block,
        "synthesis": synthesis_block,
        "artifact": artifact_block,
        "safety": {
            "raw_trace_exposed": any(key in metadata for key in _UNSAFE_TRACE_KEYS),
            "raw_content_exposed": False,
        },
    }
    return summary


def attach_lkw_proof_summary_metadata(
    metadata: dict[str, Any],
    *,
    capability: str,
) -> dict[str, Any]:
    """Attach ``lkw_proof_summary.v1`` for pipeline runs when derivable from metadata."""
    summary = build_lkw_proof_summary(metadata, capability=capability)
    if summary is not None:
        metadata[LKW_PROOF_SUMMARY_KEY] = summary
    return metadata


def _dict_metadata(metadata: dict[str, Any], key: str) -> dict[str, Any]:
    raw = metadata.get(key)
    return raw if isinstance(raw, dict) else {}


def _agent_order(app_summary: dict[str, Any]) -> list[str]:
    order: list[str] = []
    for entry in app_summary.get("agent_invocations") or []:
        if isinstance(entry, dict) and entry.get("agent_id"):
            order.append(str(entry["agent_id"]))
    return order


def _tool_calls_by_agent(app_summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in app_summary.get("agent_invocations") or []:
        if not isinstance(entry, dict):
            continue
        agent_id = entry.get("agent_id")
        raw = entry.get("total_tool_calls")
        if agent_id is not None and raw is not None:
            counts[str(agent_id)] = int(raw)
    return counts


def _diagnostic(evidence: dict[str, Any], schema_id: str) -> dict[str, Any]:
    diagnostics = evidence.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    payload = diagnostics.get(schema_id)
    return payload if isinstance(payload, dict) else {}


def _evidence_block(
    search_diag: dict[str, Any],
    synth_diag: dict[str, Any],
) -> dict[str, Any]:
    count_raw = search_diag.get("evidence_count", search_diag.get("num_results"))
    if count_raw is None:
        count_raw = synth_diag.get("source_evidence_count")
    count = int(count_raw) if count_raw is not None else 0

    source_refs = search_diag.get("source_refs")
    if source_refs is not None:
        source_refs_present: bool | None = isinstance(source_refs, list) and len(source_refs) > 0
    else:
        source_refs_present = None

    present = count >= 1 and bool(search_diag or synth_diag)
    block: dict[str, Any] = {
        "present": present,
        "count": count,
    }
    if source_refs_present is not None:
        block["source_refs_present"] = source_refs_present
    return block


def _synthesis_block(synth_diag: dict[str, Any]) -> dict[str, Any]:
    content_missing = synth_diag.get("content_missing")
    artifact_present = bool(synth_diag.get("artifact_path") or synth_diag.get("artifact_ref"))
    return {
        "shadow_write": synth_diag.get("shadow_write") is True,
        "content_missing": content_missing is True,
        "artifact_present": artifact_present,
    }


def _artifact_block(bundle: dict[str, Any], synth_diag: dict[str, Any]) -> dict[str, Any]:
    workspace = bundle.get("workspace")
    workspace_count = len(workspace) if isinstance(workspace, list) else 0
    bundle_present = workspace_count > 0 or bool(
        synth_diag.get("artifact_path") or synth_diag.get("artifact_ref")
    )
    return {
        "bundle_present": bundle_present,
        "workspace_refs_count": workspace_count,
    }


def _proof_status(
    *,
    app_summary: dict[str, Any],
    agent_order: list[str],
    tool_calls_by_agent: dict[str, int],
    evidence: dict[str, Any],
    synthesis: dict[str, Any],
    artifact: dict[str, Any],
) -> str:
    terminal_status = str(app_summary.get("terminal_status") or "").lower()
    checks = [
        terminal_status in _POSITIVE_TERMINAL_STATUSES,
        agent_order == list(_PIPELINE_AGENTS),
        all(tool_calls_by_agent.get(agent_id, 0) >= 1 for agent_id in _PIPELINE_AGENTS),
        evidence.get("present") is True and int(evidence.get("count", 0)) >= 1,
        synthesis.get("shadow_write") is True,
        synthesis.get("content_missing") is not True,
        synthesis.get("artifact_present") is True,
        artifact.get("bundle_present") is True,
    ]
    return "passed" if all(checks) else "failed"
