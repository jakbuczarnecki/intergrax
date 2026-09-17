# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project LLM spine events into read-model invocation records."""

from __future__ import annotations

from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.contracts.model_runtime_read import (
    ModelRuntimeInvocationReadResult,
    ModelRuntimeInvocationRecord,
    ModelRuntimeInvocationStatus,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def project_model_invocations_from_reconstruction(
    reconstruction: ExecutionReconstruction,
    *,
    limit: int,
) -> ModelRuntimeInvocationReadResult:
    if limit < 1:
        raise ValueError("limit must be positive")

    records: list[ModelRuntimeInvocationRecord] = []
    for positioned in reconstruction.positioned_events:
        event = positioned.event
        if event.event_type is not RuntimeEventType.LLM_CALL:
            continue
        payload = event.payload
        model_ref = str(payload.get("model") or payload.get("model_ref") or "unknown_model")
        label = str(payload.get("label") or "llm_call")
        prompt_tokens = int(payload.get("prompt_tokens") or 0)
        completion_tokens = int(payload.get("completion_tokens") or 0)
        total_tokens = int(payload.get("total_tokens") or prompt_tokens + completion_tokens)
        finish_reason = payload.get("finish_reason")
        finish = str(finish_reason).strip() if finish_reason else None
        safe_summary = sanitize_inspection_text(f"{model_ref}:{label}")
        records.append(
            ModelRuntimeInvocationRecord(
                invocation_ref=str(event.event_id),
                model_ref=model_ref,
                capability_label=label,
                invocation_status=ModelRuntimeInvocationStatus.RECORDED,
                prompt_tokens=max(prompt_tokens, 0),
                completion_tokens=max(completion_tokens, 0),
                total_tokens=max(total_tokens, 0),
                finish_reason=finish,
                tenant_id=event.tenant_id or reconstruction.tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                execution_id=event.execution_id,
                attempt_id=event.attempt_id,
                sequence_key=positioned.position.value,
                evidence_refs=(str(event.event_id),),
                safe_summary=safe_summary,
            ),
        )
        if len(records) >= limit:
            break

    sorted_records = tuple(
        sorted(records, key=lambda item: (item.sequence_key, item.invocation_ref)),
    )
    total_matching = sum(
        1
        for positioned in reconstruction.positioned_events
        if positioned.event.event_type is RuntimeEventType.LLM_CALL
    )
    return ModelRuntimeInvocationReadResult(
        records=sorted_records,
        is_truncated=total_matching > limit,
    )


__all__ = ["project_model_invocations_from_reconstruction"]
