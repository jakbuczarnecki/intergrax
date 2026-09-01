# © Artur Czarnecki. All rights reserved.

"""Emit Q2 tool-selection functional evidence from real catalog tool decisions."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.specifications.q2_tool_selection_functional_diagnostic_specification import (
    Q2_TOOL_INVOKE_OPERATION_ID,
    Q2_TOOL_QUERY_ID,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    recorder_from_exec_ctx,
    suppress_kinds_from_metadata,
)
from tool_selection_qualifier.tool_selection import ToolSelectionCandidate, artifact_ref_for_tool

_PRODUCER_COMPONENT = "agents.tool_selection_qualifier"


def emit_tool_selection_functional_evidence(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    metadata: dict[str, object],
    candidates: tuple[ToolSelectionCandidate, ...],
    selected_tool_id: str,
    invoke_succeeded: bool,
) -> None:
    recorder = recorder_from_exec_ctx(exec_ctx)
    if recorder is None:
        return

    suppressed = suppress_kinds_from_metadata(metadata)
    scope = recorder.scope_from_exec_ctx(exec_ctx)
    selected_ref = artifact_ref_for_tool(selected_tool_id)
    operation_status = (
        PipelineOperationStatus.SUCCEEDED if invoke_succeeded else PipelineOperationStatus.FAILED
    )
    recorder.record_operation_outcome(
        scope=scope,
        operation_id=Q2_TOOL_INVOKE_OPERATION_ID,
        operation_name=selected_tool_id,
        status=operation_status,
        suppressed_kinds=suppressed,
    )

    for candidate in candidates:
        ref = artifact_ref_for_tool(candidate.tool_id)
        recorder.record_candidate_rank(
            scope=scope,
            operation_id=Q2_TOOL_INVOKE_OPERATION_ID,
            query_id=Q2_TOOL_QUERY_ID,
            candidate_artifact_ref=ref,
            rank=candidate.rank,
            selected=ref == selected_ref,
            suppressed_kinds=suppressed,
        )

    recorder.record_selection(
        scope=scope,
        operation_id=Q2_TOOL_INVOKE_OPERATION_ID,
        query_id=Q2_TOOL_QUERY_ID,
        selected_artifact_ref=selected_ref,
        candidate_count=len(candidates),
        selection_reason="llm_tool_call",
        suppressed_kinds=suppressed,
    )


__all__ = ["emit_tool_selection_functional_evidence"]
