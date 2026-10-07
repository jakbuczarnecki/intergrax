# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed model-call ↔ context-assembly attribution (TRACE-X-P4-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.context.tracking.decision_evidence import compute_context_decision_evidence_fingerprint
from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.context.contracts import AssembledContext
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope
from intergrax.runtime.events.payload_registry import validate_payload_envelope
from intergrax.runtime.events.payloads.canonical import (
    ContextAssemblyPayloadV3,
    ContextAssemblyPayloadV4,
    LlmCallPayloadV2,
    LlmCallPayloadV3,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType


@dataclass(frozen=True, slots=True)
class ModelContextAttributionEvidence:
    task_id: str
    run_id: str
    attempt_id: str
    execution_id: str
    tenant_id: str | None
    model_input_messages_hash: str
    context_decision_evidence_fingerprint: str
    context_assembly_event_id: EventId
    execution_scope: ModelCallExecutionScope
    provider: str = ""
    model: str = ""


@dataclass(frozen=True, slots=True)
class ModelContextAttributionVerdict:
    attributable: bool
    reason: str = ""
    evidence: ModelContextAttributionEvidence | None = None


def _parse_context_payload(
    event: RuntimeEvent,
) -> ContextAssemblyPayloadV4 | ContextAssemblyPayloadV3 | None:
    typed = validate_payload_envelope(event.payload)
    if isinstance(typed, (ContextAssemblyPayloadV4, ContextAssemblyPayloadV3)):
        return typed
    return None


def _parse_llm_payload(event: RuntimeEvent) -> LlmCallPayloadV3 | LlmCallPayloadV2 | None:
    typed = validate_payload_envelope(event.payload)
    if isinstance(typed, (LlmCallPayloadV3, LlmCallPayloadV2)):
        return typed
    return None


def _execution_scope_from_llm(payload: LlmCallPayloadV3 | LlmCallPayloadV2) -> ModelCallExecutionScope:
    if isinstance(payload, LlmCallPayloadV3):
        return payload.execution_scope
    try:
        return ModelCallExecutionScope(payload.execution_scope)
    except ValueError:
        return ModelCallExecutionScope.PRIMARY_MODEL_CALL


def context_decision_fingerprint_from_event(event: RuntimeEvent) -> str:
    payload = _parse_context_payload(event)
    if isinstance(payload, ContextAssemblyPayloadV4):
        return payload.context_decision_evidence_fingerprint
    return ""


def try_attribute_model_call_to_context(
    context_event: RuntimeEvent,
    llm_event: RuntimeEvent,
    *,
    expected_context_fingerprint: str | None = None,
) -> ModelContextAttributionVerdict:
    if context_event.event_type != RuntimeEventType.CONTEXT_ASSEMBLED:
        return ModelContextAttributionVerdict(False, "context_event_type_mismatch")
    if llm_event.event_type != RuntimeEventType.LLM_CALL:
        return ModelContextAttributionVerdict(False, "llm_event_type_mismatch")

    context_payload = _parse_context_payload(context_event)
    llm_payload = _parse_llm_payload(llm_event)
    if context_payload is None or llm_payload is None:
        return ModelContextAttributionVerdict(False, "typed_payload_missing")

    context_fingerprint = ""
    if isinstance(context_payload, ContextAssemblyPayloadV4):
        context_fingerprint = context_payload.context_decision_evidence_fingerprint
    if not context_fingerprint:
        return ModelContextAttributionVerdict(False, "missing_decision_fingerprint")

    context_hash = context_payload.model_input_messages_hash
    llm_hash = llm_payload.model_input_messages_hash
    if not context_hash or not llm_hash or context_hash != llm_hash:
        return ModelContextAttributionVerdict(False, "model_input_hash_mismatch")

    context_event_id = validate_event_id(context_event.event_id)
    llm_context_ref = ""
    if isinstance(llm_payload, LlmCallPayloadV3):
        llm_context_ref = llm_payload.context_assembly_event_id.strip()
    if not llm_context_ref or validate_event_id(llm_context_ref) != context_event_id:
        return ModelContextAttributionVerdict(False, "context_event_id_mismatch")

    if expected_context_fingerprint is not None and expected_context_fingerprint != context_fingerprint:
        return ModelContextAttributionVerdict(False, "decision_fingerprint_contradiction")

    if (
        str(context_event.task_id) != str(llm_event.task_id)
        or str(context_event.run_id) != str(llm_event.run_id)
        or str(context_event.attempt_id) != str(llm_event.attempt_id)
        or str(context_event.execution_id) != str(llm_event.execution_id)
    ):
        return ModelContextAttributionVerdict(False, "execution_identity_mismatch")

    left_tenant = context_event.tenant_id
    right_tenant = llm_event.tenant_id
    if left_tenant != right_tenant:
        return ModelContextAttributionVerdict(False, "tenant_mismatch")

    scope = _execution_scope_from_llm(llm_payload)
    evidence = ModelContextAttributionEvidence(
        task_id=str(llm_event.task_id),
        run_id=str(llm_event.run_id),
        attempt_id=str(llm_event.attempt_id),
        execution_id=str(llm_event.execution_id),
        tenant_id=llm_event.tenant_id,
        model_input_messages_hash=llm_hash,
        context_decision_evidence_fingerprint=context_fingerprint,
        context_assembly_event_id=context_event_id,
        execution_scope=scope,
        provider=llm_payload.provider,
        model=llm_payload.model,
    )
    return ModelContextAttributionVerdict(True, evidence=evidence)


def recompute_context_decision_fingerprint(assembled: AssembledContext) -> str:
    return compute_context_decision_evidence_fingerprint(assembled)


__all__ = [
    "ModelContextAttributionEvidence",
    "ModelContextAttributionVerdict",
    "context_decision_fingerprint_from_event",
    "recompute_context_decision_fingerprint",
    "try_attribute_model_call_to_context",
]
