# © Artur Czarnecki. All rights reserved.

"""Emit C1 synthesis functional evidence from local_synthesizer output."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    C1_RAG_EXPECTED_SELECTION_ARTIFACT,
    C1_RAG_SYNTHESIZE_OPERATION_ID,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    recorder_from_exec_ctx,
    suppress_kinds_from_metadata,
)


def emit_synthesize_functional_evidence(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    metadata: dict[str, Any],
    selected_artifact_ref: str | None,
    output_artifact_ref: str,
    synthesize_succeeded: bool,
) -> dict[str, object] | None:
    recorder = recorder_from_exec_ctx(exec_ctx)
    if recorder is None:
        return None

    suppressed = suppress_kinds_from_metadata(metadata)
    scope = recorder.scope_from_exec_ctx(exec_ctx)
    resolved_selection = selected_artifact_ref or C1_RAG_EXPECTED_SELECTION_ARTIFACT
    if synthesize_succeeded:
        recorder.record_output_relation(
            scope=scope,
            operation_id=C1_RAG_SYNTHESIZE_OPERATION_ID,
            selected_artifact_ref=resolved_selection,
            output_artifact_ref=output_artifact_ref,
            relation_kind="synthesized_from",
            suppressed_kinds=suppressed,
        )
    return {
        "selected_artifact_ref": resolved_selection,
        "output_artifact_ref": output_artifact_ref,
        "synthesize_succeeded": synthesize_succeeded,
    }


__all__ = ["emit_synthesize_functional_evidence"]
