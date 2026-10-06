# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4 / P4-R1 qualification support (model ↔ context attribution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.runtime.events.payload_registry import validate_payload_envelope
from intergrax.runtime.events.payloads.canonical import (
    ContextAssemblyPayloadV3,
    ContextAssemblyPayloadV4,
    LlmCallPayloadV2,
    LlmCallPayloadV3,
)
from intergrax.runtime.llm.model_context_attribution import (
    ModelContextAttributionVerdict,
    try_attribute_model_call_to_context,
)

TRACE_X_P4_START_HEAD: Final[str] = "7d782af85fa97807b882cc9063a5d57aef899e57"


@dataclass(frozen=True, slots=True)
class P4GateEvidence:
    gate_id: str
    description: str
    nodeids: tuple[str, ...]


P4_R1_GATE_REGISTRY: tuple[P4GateEvidence, ...] = (
    P4GateEvidence("TXP4R1-Q01", "START_HEAD ancestry", ("test_txp4_q01_start_head_ancestry",)),
    P4GateEvidence(
        "TXP4R1-Q04",
        "recorder binding resets recorder and tenant",
        ("test_txp4r1_q04_recorder_binding_resets_tenant",),
    ),
    P4GateEvidence(
        "TXP4R1-Q05",
        "nested tenant binding restores outer tenant",
        ("test_txp4r1_q05_nested_tenant_binding",),
    ),
    P4GateEvidence(
        "TXP4R1-Q06",
        "sequential tenant executions do not leak",
        ("test_txp4r1_q06_sequential_tenant_isolation",),
    ),
    P4GateEvidence(
        "TXP4R1-Q08",
        "invalid execution_scope rejected",
        ("test_txp4r1_q08_invalid_execution_scope_rejected",),
    ),
    P4GateEvidence(
        "TXP4R1-Q11",
        "different decision chains differ despite same messages",
        ("test_txp4r1_q11_same_messages_different_decision_fingerprint",),
    ),
    P4GateEvidence(
        "TXP4R1-Q14",
        "positive context-decision to model E2E",
        ("test_txp4_q12_primary_context_model_e2e_attribution",),
    ),
    P4GateEvidence(
        "TXP4R1-Q20",
        "cross-tenant attribution rejected",
        ("test_txp4r1_q20_cross_tenant_attribution_rejected",),
    ),
    P4GateEvidence(
        "TXP4R1-Q25",
        "streaming not production primary",
        ("test_txp4r1_q25_streaming_not_production_primary",),
    ),
)


def parse_context_assembly_payload(
    event: RuntimeEvent,
) -> ContextAssemblyPayloadV4 | ContextAssemblyPayloadV3 | None:
    typed = validate_payload_envelope(event.payload)
    if isinstance(typed, (ContextAssemblyPayloadV4, ContextAssemblyPayloadV3)):
        return typed
    return None


def parse_llm_call_payload(event: RuntimeEvent) -> LlmCallPayloadV3 | LlmCallPayloadV2 | None:
    typed = validate_payload_envelope(event.payload)
    if isinstance(typed, (LlmCallPayloadV3, LlmCallPayloadV2)):
        return typed
    return None


def attribute_model_call_to_context(
    context_event: RuntimeEvent,
    llm_event: RuntimeEvent,
    *,
    expected_context_fingerprint: str | None = None,
) -> ModelContextAttributionVerdict:
    return try_attribute_model_call_to_context(
        context_event,
        llm_event,
        expected_context_fingerprint=expected_context_fingerprint,
    )


__all__ = [
    "P4GateEvidence",
    "P4_R1_GATE_REGISTRY",
    "TRACE_X_P4_START_HEAD",
    "attribute_model_call_to_context",
    "parse_context_assembly_payload",
    "parse_llm_call_payload",
]
