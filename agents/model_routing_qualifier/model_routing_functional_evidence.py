# © Artur Czarnecki. All rights reserved.

"""Emit Q4 model-routing functional evidence from real routing decisions."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.specifications.q4_model_routing_functional_diagnostic_specification import (
    Q4_MODEL_GENERATE_OPERATION_ID,
    Q4_MODEL_QUERY_ID,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    recorder_from_exec_ctx,
    suppress_kinds_from_metadata,
)
from model_routing_qualifier.model_routing import RoutingProfileCandidate, artifact_ref_for_profile

_PRODUCER_COMPONENT = "agents.model_routing_qualifier"


def emit_model_routing_functional_evidence(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    metadata: Mapping[str, object],
    candidates: tuple[RoutingProfileCandidate, ...],
    selected_profile_ref: str,
    invoke_succeeded: bool,
    raw_model_output: str,
) -> None:
    recorder = recorder_from_exec_ctx(exec_ctx)
    if recorder is None:
        return

    suppressed = suppress_kinds_from_metadata(metadata)
    scope = recorder.scope_from_exec_ctx(exec_ctx)
    operation_status = (
        PipelineOperationStatus.SUCCEEDED if invoke_succeeded else PipelineOperationStatus.FAILED
    )
    bounded_output = raw_model_output[:500]
    recorder.record_operation_outcome(
        scope=scope,
        operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
        operation_name=selected_profile_ref,
        status=operation_status,
        suppressed_kinds=suppressed,
    )

    for candidate in candidates:
        ref = artifact_ref_for_profile(candidate.profile)
        recorder.record_candidate_rank(
            scope=scope,
            operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
            query_id=Q4_MODEL_QUERY_ID,
            candidate_artifact_ref=ref,
            rank=candidate.rank,
            selected=ref == selected_profile_ref,
            suppressed_kinds=suppressed,
        )

    recorder.record_selection(
        scope=scope,
        operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
        query_id=Q4_MODEL_QUERY_ID,
        selected_artifact_ref=selected_profile_ref,
        candidate_count=len(candidates),
        selection_reason="llm_routing_evaluator",
        suppressed_kinds=suppressed,
    )

    if invoke_succeeded and bounded_output:
        recorder.record_output_relation(
            scope=scope,
            operation_id=Q4_MODEL_GENERATE_OPERATION_ID,
            selected_artifact_ref=selected_profile_ref,
            output_artifact_ref=f"text:{bounded_output}",
            relation_kind="model_response",
            suppressed_kinds=suppressed,
        )


__all__ = ["emit_model_routing_functional_evidence"]
