# © Artur Czarnecki. All rights reserved.

"""Emit C1 RAG functional evidence from local_search retrieval results."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from local_search.retrieval_selection import (
    SearchRetrievalCandidate,
    artifact_ref_from_candidate,
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
    candidates: tuple[SearchRetrievalCandidate, ...],
    selected_artifact_ref: str,
    retrieve_succeeded: bool,
) -> None:
    """Record retrieval operation, candidates, and selection facts — observation only."""
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

    for index, candidate in enumerate(candidates, start=1):
        ref = artifact_ref_from_candidate(candidate)
        recorder.record_candidate_rank(
            scope=scope,
            operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
            query_id=C1_RAG_QUERY_ID,
            candidate_artifact_ref=ref,
            rank=index,
            selected=ref == selected_artifact_ref,
            score=candidate.score,
            suppressed_kinds=suppressed,
        )

    recorder.record_selection(
        scope=scope,
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        query_id=C1_RAG_QUERY_ID,
        selected_artifact_ref=selected_artifact_ref,
        candidate_count=len(candidates),
        selection_reason="top_ranked",
        suppressed_kinds=suppressed,
    )


__all__ = ["emit_search_functional_evidence"]
