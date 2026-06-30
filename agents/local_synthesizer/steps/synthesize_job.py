# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from intergrax.agents.authoring.runtime_tool_helpers import (
    exec_ctx_from_step,
    invoke_catalog_tool,
    request_metadata,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import WORKSPACE_WRITE_FILE_TOOL_ID

SYNTHESIZE_STEP_ID = "local_synthesizer_step"
_DEFAULT_OUTPUT_NAME = "draft.md"
_PRIOR_OUTPUTS_KEY = "prior_agent_outputs"
_SHARED_CONTEXT_READS_KEY = "shared_context_reads"
_SEARCH_AGENT_ID = "local_search"


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _prior_output_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in (_PRIOR_OUTPUTS_KEY, _SHARED_CONTEXT_READS_KEY):
        raw = metadata.get(key)
        if not isinstance(raw, dict):
            continue
        for value in raw.values():
            if isinstance(value, dict):
                entries.append(value)
    return entries


def _search_summary_from_prior_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    agent_id = entry.get("agent_id")
    if agent_id is not None and str(agent_id) != _SEARCH_AGENT_ID:
        return None
    structured = _as_dict(entry.get("structured_data"))
    summary = structured.get("search_summary")
    return dict(summary) if isinstance(summary, dict) else None


def _merge_pipeline_search_handoff(metadata: dict[str, Any]) -> dict[str, Any]:
    """Lift search evidence from platform prior-output / shared-context reads."""
    if metadata.get("evidence"):
        return metadata
    existing_summary = metadata.get("search_summary")
    if isinstance(existing_summary, dict) and existing_summary.get("evidence"):
        return metadata

    merged = dict(metadata)
    for entry in _prior_output_entries(metadata):
        search_summary = _search_summary_from_prior_entry(entry)
        if not search_summary:
            continue
        if "search_summary" not in merged:
            merged["search_summary"] = dict(search_summary)
        evidence = search_summary.get("evidence")
        if isinstance(evidence, list) and evidence and not merged.get("evidence"):
            merged["evidence"] = [item for item in evidence if isinstance(item, dict)]
        break
    return merged


def _parse_evidence(metadata: dict[str, Any]) -> list[dict[str, object]]:
    raw = metadata.get("evidence")
    if raw is None:
        search_summary = metadata.get("search_summary")
        if isinstance(search_summary, dict):
            nested = search_summary.get("evidence")
            if isinstance(nested, list):
                raw = nested
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _resolve_output_name(metadata: dict[str, Any]) -> str:
    raw = metadata.get("output_name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _DEFAULT_OUTPUT_NAME


def _build_draft_from_evidence(metadata: dict[str, Any], evidence: list[dict[str, object]]) -> str:
    lines = ["# Synthesis draft", ""]
    search_summary = metadata.get("search_summary")
    if isinstance(search_summary, dict):
        query = search_summary.get("query")
        if isinstance(query, str) and query.strip():
            lines.extend([f"Query: {query.strip()}", ""])
    elif isinstance(search_summary, str) and search_summary.strip():
        lines.extend([search_summary.strip(), ""])

    for index, item in enumerate(evidence, start=1):
        lines.append(f"## Evidence {index}")
        source_path = item.get("source_path") or item.get("source")
        if isinstance(source_path, str) and source_path.strip():
            lines.append(f"Source: {source_path.strip()}")
        text = item.get("text") or item.get("content")
        if isinstance(text, str) and text.strip():
            lines.append(text.strip())
        lines.append("")

    return "\n".join(lines).strip()


def _resolve_content(
    step_ctx: AgentStepContext,
    metadata: dict[str, Any],
    evidence: list[dict[str, object]],
) -> str:
    draft = metadata.get("draft")
    if isinstance(draft, str) and draft.strip():
        return draft.strip()
    if evidence:
        return _build_draft_from_evidence(metadata, evidence)
    message = str(step_ctx.message or metadata.get("message") or "").strip()
    if message:
        return f"# Synthesis draft\n\n{message}\n"
    return ""


def _content_type_for_path(path: str) -> str:
    if path.endswith(".md"):
        return "text/markdown"
    return "text/plain"


def _artifact_ref_from_tool_entry(entry: dict[str, Any]) -> str | None:
    artifact_id = entry.get("artifact_id")
    workspace_id = entry.get("workspace_id")
    if artifact_id and workspace_id:
        return f"{workspace_id}/{artifact_id}"
    if artifact_id:
        return str(artifact_id)
    return None


def _output(
    *,
    run_id: str,
    used: bool,
    reason: str,
    output_name: str | None = None,
    artifact_path: str | None = None,
    artifact_ref: str | None = None,
    shadow_workspace: bool = False,
    num_evidence_items: int = 0,
    raw_tool_reason: str | None = None,
) -> dict[str, object]:
    if used:
        answer = f"local_synthesizer: synthesize complete — {output_name or _DEFAULT_OUTPUT_NAME}"
    else:
        answer = f"local_synthesizer: synthesize failed — {reason}"
    summary: dict[str, object] = {
        "used": used,
        "reason": reason,
        "output_name": output_name,
        "artifact_path": artifact_path,
        "artifact_ref": artifact_ref,
        "shadow_workspace": shadow_workspace,
        "num_evidence_items": num_evidence_items,
    }
    if raw_tool_reason:
        summary["raw_tool_reason"] = raw_tool_reason
    return {
        "summary": answer,
        "answer": answer,
        "run_id": run_id,
        "synthesize_summary": summary,
    }


def _task_metadata(step_ctx: AgentStepContext, exec_ctx) -> dict[str, Any]:
    metadata = dict(request_metadata(exec_ctx))
    for key in (
        "shadow_workspace",
        "evidence",
        "draft",
        "output_name",
        "search_summary",
        "message",
        _PRIOR_OUTPUTS_KEY,
        _SHARED_CONTEXT_READS_KEY,
    ):
        if key not in metadata:
            value = step_ctx.metadata.get(key)
            if value is not None:
                metadata[key] = value
    return _merge_pipeline_search_handoff(metadata)


async def run_synthesize_job(step_ctx: AgentStepContext) -> dict[str, object]:
    """LKW.1.3 — minimal shadow draft via workspace.write_file."""
    exec_ctx = exec_ctx_from_step(step_ctx)
    metadata = _task_metadata(step_ctx, exec_ctx)
    shadow_workspace = bool(metadata.get("shadow_workspace"))
    evidence = _parse_evidence(metadata)
    output_name = _resolve_output_name(metadata)
    num_evidence_items = len(evidence)

    if not shadow_workspace:
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason="shadow_workspace_required",
            output_name=output_name,
            num_evidence_items=num_evidence_items,
        )

    content = _resolve_content(step_ctx, metadata, evidence)
    if not content:
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason="content_missing",
            output_name=output_name,
            shadow_workspace=True,
            num_evidence_items=num_evidence_items,
        )

    if exec_ctx is None:
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason="tool_gateway_not_available",
            output_name=output_name,
            shadow_workspace=True,
            num_evidence_items=num_evidence_items,
        )

    entry = await invoke_catalog_tool(
        exec_ctx,
        tool_name=WORKSPACE_WRITE_FILE_TOOL_ID,
        agent_id=step_ctx.agent_id,
        step_id=SYNTHESIZE_STEP_ID,
        tool_input={
            "path": output_name,
            "content": content,
            "content_type": _content_type_for_path(output_name),
        },
    )

    if entry.get("status") != "success":
        tool_reason = entry.get("reason")
        return _output(
            run_id=step_ctx.run_id,
            used=False,
            reason="write_failed",
            output_name=output_name,
            shadow_workspace=True,
            num_evidence_items=num_evidence_items,
            raw_tool_reason=str(tool_reason) if tool_reason else None,
        )

    artifact_path = entry.get("relative_path") or output_name
    return _output(
        run_id=step_ctx.run_id,
        used=True,
        reason="write_complete",
        output_name=output_name,
        artifact_path=str(artifact_path),
        artifact_ref=_artifact_ref_from_tool_entry(entry),
        shadow_workspace=True,
        num_evidence_items=num_evidence_items,
    )
