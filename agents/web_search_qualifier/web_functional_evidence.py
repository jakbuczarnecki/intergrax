# © Artur Czarnecki. All rights reserved.

"""Emit Q3 web-search functional evidence from real provider decisions."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.specifications.q3_web_search_functional_diagnostic_specification import (
    Q3_WEB_EXTRACT_OPERATION_ID,
    Q3_WEB_QUERY_CONSTRUCT_OPERATION_ID,
    Q3_WEB_QUERY_ID,
    Q3_WEB_SEARCH_OPERATION_ID,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    recorder_from_exec_ctx,
    suppress_kinds_from_metadata,
)
from web_search_qualifier.url_identity import artifact_ref_for_url
from web_search_qualifier.web_search import WebSearchCandidate

_PRODUCER_COMPONENT = "agents.web_search_qualifier"


def emit_web_search_functional_evidence(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    metadata: dict[str, object],
    actual_query: str,
    search_succeeded: bool,
    candidates: tuple[WebSearchCandidate, ...],
    selected_url: str,
    extracted_fact: str,
    selection_mode: str | None = None,
    policy_id: str | None = None,
    raw_selector_response: str | None = None,
) -> None:
    recorder = recorder_from_exec_ctx(exec_ctx)
    if recorder is None:
        return

    suppressed = suppress_kinds_from_metadata(metadata)
    scope = recorder.scope_from_exec_ctx(exec_ctx)
    selected_ref = artifact_ref_for_url(selected_url)
    search_status = (
        PipelineOperationStatus.SUCCEEDED if search_succeeded else PipelineOperationStatus.FAILED
    )

    recorder.record_operation_outcome(
        scope=scope,
        operation_id=Q3_WEB_QUERY_CONSTRUCT_OPERATION_ID,
        operation_name=actual_query[:500],
        status=PipelineOperationStatus.SUCCEEDED,
        suppressed_kinds=suppressed,
    )
    recorder.record_operation_outcome(
        scope=scope,
        operation_id=Q3_WEB_SEARCH_OPERATION_ID,
        operation_name=actual_query[:500],
        status=search_status,
        suppressed_kinds=suppressed,
    )

    for candidate in candidates:
        ref = artifact_ref_for_url(candidate.url)
        recorder.record_candidate_rank(
            scope=scope,
            operation_id=Q3_WEB_SEARCH_OPERATION_ID,
            query_id=Q3_WEB_QUERY_ID,
            candidate_artifact_ref=ref,
            rank=candidate.rank,
            selected=ref == selected_ref,
            suppressed_kinds=suppressed,
        )

    selection_reason = _selection_reason(
        selection_mode=selection_mode,
        policy_id=policy_id,
    )
    recorder.record_selection(
        scope=scope,
        operation_id=Q3_WEB_SEARCH_OPERATION_ID,
        query_id=Q3_WEB_QUERY_ID,
        selected_artifact_ref=selected_ref,
        candidate_count=len(candidates),
        selection_reason=selection_reason,
        suppressed_kinds=suppressed,
    )

    if extracted_fact.strip():
        recorder.record_output_relation(
            scope=scope,
            operation_id=Q3_WEB_EXTRACT_OPERATION_ID,
            selected_artifact_ref=selected_ref,
            output_artifact_ref=f"fact:{extracted_fact.strip()[:200]}",
            relation_kind="extracted_from",
            suppressed_kinds=suppressed,
        )


def _selection_reason(*, selection_mode: str | None, policy_id: str | None) -> str:
    if selection_mode == "policy" and policy_id:
        return f"policy:{policy_id}"
    if selection_mode == "llm":
        return "llm_source_selection"
    return "llm_source_selection"


__all__ = ["emit_web_search_functional_evidence"]
