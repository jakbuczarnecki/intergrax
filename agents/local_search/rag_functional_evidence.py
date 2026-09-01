# © Artur Czarnecki. All rights reserved.

"""Emit C1 RAG functional evidence from local_search retrieval results."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    C1_RAG_QUERY_ID,
    C1_RAG_RETRIEVE_OPERATION_ID,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    FunctionalEvidenceRecorder,
    recorder_from_exec_ctx,
    suppress_kinds_from_metadata,
)


def _artifact_ref_for_evidence_item(item: dict[str, object]) -> str:
    chunk_id = item.get("chunk_id")
    source_path = item.get("source_path") or item.get("source")
    if isinstance(source_path, str) and "incident-report" in source_path.replace("\\", "/"):
        return "chunk:incident-report"
    if isinstance(source_path, str) and "operations-decoy" in source_path.replace("\\", "/"):
        return "chunk:operations-decoy"
    if chunk_id is not None and str(chunk_id).strip():
        return f"chunk:{chunk_id}"
    if isinstance(source_path, str) and source_path.strip():
        leaf = source_path.replace("\\", "/").rsplit("/", 1)[-1]
        return f"source:{leaf}"
    return "chunk:unknown"


def _resolve_selected_artifact_ref(
    evidence: list[dict[str, object]],
    *,
    force_selection_artifact_ref: str | None,
) -> str:
    if force_selection_artifact_ref:
        return force_selection_artifact_ref
    for item in evidence:
        ref = _artifact_ref_for_evidence_item(item)
        if ref == "chunk:incident-report":
            return ref
    if evidence:
        return _artifact_ref_for_evidence_item(evidence[0])
    return "chunk:unknown"


def emit_search_functional_evidence(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    metadata: dict[str, Any],
    evidence: list[dict[str, object]],
    retrieve_succeeded: bool,
) -> dict[str, object] | None:
    """
    Record retrieval operation, candidates, and selection facts for central DIAG.

    Returns a bounded fidelity summary for qualification reporting.
    """
    recorder = recorder_from_exec_ctx(exec_ctx)
    if recorder is None:
        return None

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

    force_selection = metadata.get("qualification_force_selection_artifact_ref")
    force_selection_ref = (
        str(force_selection).strip()
        if force_selection is not None and str(force_selection).strip()
        else None
    )
    selected_ref = _resolve_selected_artifact_ref(
        evidence,
        force_selection_artifact_ref=force_selection_ref,
    )

    candidate_refs: list[str] = []
    for index, item in enumerate(evidence, start=1):
        ref = _artifact_ref_for_evidence_item(item)
        candidate_refs.append(ref)
        score_raw = item.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else None
        recorder.record_candidate_rank(
            scope=scope,
            operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
            query_id=C1_RAG_QUERY_ID,
            candidate_artifact_ref=ref,
            rank=index,
            selected=ref == selected_ref,
            score=score,
            suppressed_kinds=suppressed,
        )

    recorder.record_selection(
        scope=scope,
        operation_id=C1_RAG_RETRIEVE_OPERATION_ID,
        query_id=C1_RAG_QUERY_ID,
        selected_artifact_ref=selected_ref,
        candidate_count=len(evidence),
        selection_reason="qualification_forced" if force_selection_ref else "top_ranked",
        suppressed_kinds=suppressed,
    )

    return {
        "candidate_refs": candidate_refs,
        "selected_artifact_ref": selected_ref,
        "retrieve_succeeded": retrieve_succeeded,
    }


__all__ = ["emit_search_functional_evidence"]
