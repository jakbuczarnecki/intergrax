# © Artur Czarnecki. All rights reserved.

"""Local workspace run response metadata enrichment (LKW evidence read model)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.runtime.task.task import TaskResult
from intergrax.runtime.task.task_metadata_keys import TaskResultMetadataKey
from lkw_shared.evidence_slice import (
    LKW_EVIDENCE_SCHEMA_VERSION,
    build_lkw_evidence_slice_from_step_diagnostics,
)


def attach_lkw_evidence_metadata(
    metadata: dict[str, Any],
    *,
    task_result: TaskResult,
    capability: str | None = None,
) -> dict[str, Any]:
    """Attach curated ``lkw_evidence.v1`` slice derived from execution trace diagnostics."""
    step_diagnostics = _step_diagnostics_from_task_result(task_result)
    terminal_status = _terminal_status_from_metadata(metadata)
    evidence = build_lkw_evidence_slice_from_step_diagnostics(
        step_diagnostics,
        capability=capability,
        agent_id=task_result.agent_id,
        run_id=task_result.run_id,
        task_id=task_result.task_id,
        terminal_status=terminal_status,
    )
    metadata[LKW_EVIDENCE_SCHEMA_VERSION] = evidence.model_dump(mode="json")
    return metadata


def _step_diagnostics_from_task_result(task_result: TaskResult) -> dict[str, Any]:
    execution = task_result.execution_result
    if execution is None:
        return {}
    trace_summary = execution.structured_data.get(AcpStructuredDataKey.TRACE_SUMMARY)
    if not isinstance(trace_summary, dict):
        return {}
    raw = trace_summary.get("step_diagnostics")
    return raw if isinstance(raw, dict) else {}


def _terminal_status_from_metadata(metadata: dict[str, Any]) -> str | None:
    app_summary = metadata.get(TaskResultMetadataKey.APPLICATION_RUN_SUMMARY)
    if isinstance(app_summary, dict):
        terminal_status = app_summary.get("terminal_status")
        if terminal_status is not None:
            return str(terminal_status)
    return None
