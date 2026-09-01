# © Artur Czarnecki. All rights reserved.

"""Emit C1 synthesis functional evidence from local_synthesizer output."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.diagnostics.specifications.c1_rag_functional_diagnostic_specification import (
    C1_RAG_SYNTHESIZE_OPERATION_ID,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    recorder_from_exec_ctx,
    suppress_kinds_from_metadata,
)


def emit_synthesize_functional_evidence(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    metadata: dict[str, object],
    selected_artifact_ref: str,
    output_artifact_ref: str,
    synthesize_succeeded: bool,
) -> None:
    recorder = recorder_from_exec_ctx(exec_ctx)
    if recorder is None:
        return

    suppressed = suppress_kinds_from_metadata(metadata)
    scope = recorder.scope_from_exec_ctx(exec_ctx)
    if synthesize_succeeded:
        recorder.record_output_relation(
            scope=scope,
            operation_id=C1_RAG_SYNTHESIZE_OPERATION_ID,
            selected_artifact_ref=selected_artifact_ref,
            output_artifact_ref=output_artifact_ref,
            relation_kind="synthesized_from",
            suppressed_kinds=suppressed,
        )


__all__ = ["emit_synthesize_functional_evidence"]
