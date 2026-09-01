# © Artur Czarnecki. All rights reserved.

"""Emit C1 RAG functional evidence from local_search retrieval results."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.c1_retrieval_evidence import (
    RetrievalEvidenceItem,
    artifact_ref_from_retrieval_item,
)
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    C1_RAG_QUERY_ID,
    C1_RAG_RETRIEVE_OPERATION_ID,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    recorder_from_exec_ctx,
    suppress_kinds_from_metadata,
)


def emit_search_functional_evidence(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    metadata: dict[str, object],
    evidence_items: tuple[RetrievalEvidenceItem, ...],
    actual_selected_artifact_ref: str,
    retrieve_succeeded: bool,
) -> None:
    """
    Record retrieval operation, candidates, and selection facts for central DIAG.

    ``actual_selected_artifact_ref`` must come from the pipeline decision point;
    instrumentation only observes and persists that fact.
    """
    recorder = recorder_from_exec_ctx(exec_ctx)
    if recorder is None:
        return

    suppressed = suppress_kinds_from_metadata(metadata)
    scope = recorder.scope_from_exec_ctx(exec_ctx)
    operation_status = (
        PipelineOperationStatus.SUCCEEDED if retrieve_succeeded else PipelineOperationStatus.FAILED
    )
    recorder.record_operation_outcome(
        scope=scope,
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        operation_name="rag.retrieve",
        status=operation_status,
        suppressed_kinds=suppressed,
    )

    for index, item in enumerate(evidence_items, start=1):
        ref = artifact_ref_from_retrieval_item(item)
        recorder.record_candidate_rank(
            scope=scope,
            operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
            query_id=C1_RAG_QUERY_ID,
            candidate_artifact_ref=ref,
            rank=index,
            selected=ref == actual_selected_artifact_ref,
            score=item.score,
            suppressed_kinds=suppressed,
        )

    recorder.record_selection(
        scope=scope,
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        query_id=C1_RAG_QUERY_ID,
        selected_artifact_ref=actual_selected_artifact_ref,
        candidate_count=len(evidence_items),
        selection_reason="top_ranked",
        suppressed_kinds=suppressed,
    )


__all__ = ["emit_search_functional_evidence"]
