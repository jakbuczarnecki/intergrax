# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical RuntimeEvent LLM_CALL recording (TRACE-X-P4)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.execution_identity import validate_event_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_recording import RuntimeEventRecorderPort
from intergrax.runtime.llm.model_call_attribution import (
    get_model_call_execution_scope,
    peek_model_call_attribution_ids,
    peek_pending_context_assembly_event_id,
)
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope
from intergrax.runtime.events.active_runtime_event_recorder import (
    peek_active_runtime_event_recorder,
    peek_active_runtime_event_tenant_id,
)
from intergrax.runtime.events.context_skill_recording import _canonical_event_identity
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads.canonical import LlmCallPayloadV3
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


def record_llm_call_runtime_event(
    bus: RuntimeEventRecorderPort,
    *,
    task_id: str,
    run_id: str,
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    finish_reason: str | None,
    model_input_messages_hash: str,
    execution_scope: ModelCallExecutionScope,
    tenant_id: str | None = None,
    context_assembly_event_id: str | None = None,
    node_id: str = "",
    agent_id: str | None = None,
    step_id: str = "",
    label: str = "",
) -> None:
    if not model_input_messages_hash:
        raise ValueError("model_input_messages_hash required for LLM_CALL evidence")
    resolved_task_id, resolved_run_id, attempt_id, execution_id = _canonical_event_identity(
        task_id=task_id,
        run_id=run_id,
    )
    resolved_tenant = tenant_id if tenant_id else peek_active_runtime_event_tenant_id() or None
    node_id_attr, agent_id_attr, step_id_attr, label_attr = peek_model_call_attribution_ids()
    resolved_node_id = node_id or node_id_attr
    resolved_agent_id = agent_id if agent_id is not None else (agent_id_attr or None)
    resolved_step_id = step_id or step_id_attr
    resolved_label = label or label_attr
    resolved_context_event_id = (context_assembly_event_id or peek_pending_context_assembly_event_id()).strip()
    if execution_scope == ModelCallExecutionScope.PRIMARY_MODEL_CALL and not resolved_context_event_id:
        raise ValueError("context_assembly_event_id required for PRIMARY_MODEL_CALL LLM_CALL evidence")
    if resolved_context_event_id:
        validate_event_id(resolved_context_event_id)
    promote: dict[str, Any] = {
        "model": model,
        "provider": provider,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model_input_messages_hash": model_input_messages_hash,
        "execution_scope": execution_scope.value,
    }
    if resolved_context_event_id:
        promote["context_assembly_event_id"] = resolved_context_event_id
    if finish_reason is not None:
        promote["finish_reason"] = finish_reason
    bus.record(
        runtime_event_with_payload(
            RuntimeEvent(
                tenant_id=resolved_tenant,
                task_id=resolved_task_id,
                run_id=resolved_run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                node_id=resolved_node_id,
                agent_id=resolved_agent_id,
                step_id=resolved_step_id,
                event_type=RuntimeEventType.LLM_CALL,
                phase=ExecutionPhase.STEP_EXECUTION,
                correlation_id=task_id,
            ),
            LlmCallPayloadV3(
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                label=resolved_label,
                model_input_messages_hash=model_input_messages_hash,
                execution_scope=execution_scope,
                context_assembly_event_id=resolved_context_event_id,
            ),
            promote_fields=promote,
        )
    )


def maybe_record_llm_call_from_usage_end(
    *,
    run_id: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    success: bool,
    finish_reason: str | None,
    model_input_messages_hash: str,
) -> None:
    if not success:
        return
    bus = peek_active_runtime_event_recorder()
    if bus is None:
        return
    if not model_input_messages_hash:
        return
    from intergrax.contracts.execution_identity import peek_active_execution_identity
    from intergrax.runtime.execution.failure_evidence.active_context import (
        peek_active_execution_evidence_context,
    )

    active = peek_active_execution_identity()
    if active is None:
        return
    _active_run_id, _attempt_id = active
    evidence = peek_active_execution_evidence_context()
    if evidence is None:
        return
    record_llm_call_runtime_event(
        bus,
        task_id=str(evidence.task_id),
        run_id=run_id or _active_run_id,
        tenant_id=evidence.tenant_id,
        model=model,
        provider=provider,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        finish_reason=finish_reason,
        model_input_messages_hash=model_input_messages_hash,
        execution_scope=get_model_call_execution_scope(),
    )


__all__ = [
    "maybe_record_llm_call_from_usage_end",
    "record_llm_call_runtime_event",
]
