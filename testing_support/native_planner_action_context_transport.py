# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — live transport qualification harness for typed planner annotations."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.tools.investigation_proof import (
    build_completed_observation_reference_index,
    format_investigation_follow_up_context,
)
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    PLANNER_ACTION_CONTEXT_TOOL_ID,
    NativePlannerProtocolConfig,
    NativePlannerProtocolMode,
    append_planner_action_context_schema,
    resolve_native_planner_protocol,
)

_LIVE_FLAG = "INTERGRAX_DS_E2E_12_LIVE"
_REQUIRED_QWEN_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class PlannerTransportCaptureRecord:
    """Sanitized capture for one PoC planner transport attempt."""

    model: str | None
    provider: str | None
    tool_call_names: tuple[str, ...]
    annotation_count: int
    business_call_count: int
    basis_refs: tuple[str, ...]
    purpose_present: bool
    transport_success: bool
    response_id: str | None


@dataclass(frozen=True, slots=True)
class PlannerTransportQualificationResult:
    """Aggregate outcome for one provider qualification run."""

    provider: str
    model: str | None
    required_attempts: int
    successful_attempts: int
    captures: tuple[PlannerTransportCaptureRecord, ...]
    gate_passed: bool


def live_transport_enabled() -> bool:
    return os.environ.get(_LIVE_FLAG, "").strip() == "1"


def _probe_business_tool_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "probe.fetch_logs",
            "description": "Fetch recent application logs for investigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["service"],
                "additionalProperties": False,
            },
        },
    }


def _follow_up_messages(
    *,
    available_evidence_references: Sequence[str],
) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You are an incident investigator. When prior evidence exists, you must "
                "call the reserved planner annotation tool and at least one business tool "
                "in the same response."
            ),
        ),
        ChatMessage(
            role="user",
            content="Investigate elevated error rate for checkout service.",
        ),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_prior_probe",
                    "type": "function",
                    "function": {
                        "name": "probe.fetch_logs",
                        "arguments": json.dumps({"service": "checkout", "limit": 5}),
                    },
                }
            ],
        ),
        ChatMessage(
            role="tool",
            content=json.dumps(
                {
                    "evidence_reference": available_evidence_references[0],
                    "payload": {"errors": 42},
                }
            ),
            tool_call_id="call_prior_probe",
        ),
        ChatMessage(
            role="system",
            content=format_investigation_follow_up_context(
                round_index=2,
                available_evidence_references=available_evidence_references,
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Continue investigation. In one response, emit exactly one "
                f"{PLANNER_ACTION_CONTEXT_TOOL_ID} call with a valid basis reference "
                "and purpose, plus at least one probe.fetch_logs business tool call."
            ),
        ),
    ]


def _capture_from_response(
    response: LLMAdapterResponse,
    *,
    transport_success: bool,
    basis_refs: tuple[str, ...] = (),
    purpose_present: bool = False,
) -> PlannerTransportCaptureRecord:
    names = tuple(call.name for call in response.tool_calls)
    annotation_count = sum(1 for name in names if name == PLANNER_ACTION_CONTEXT_TOOL_ID)
    business_count = len(names) - annotation_count
    return PlannerTransportCaptureRecord(
        model=response.model,
        provider=response.provider,
        tool_call_names=names,
        annotation_count=annotation_count,
        business_call_count=business_count,
        basis_refs=basis_refs,
        purpose_present=purpose_present,
        transport_success=transport_success,
        response_id=response.response_id,
    )


def run_one_planner_transport_attempt(
    adapter: LLMAdapter,
    *,
    available_evidence_references: Sequence[str] = ("observation.probe.fetch_logs.prior",),
) -> PlannerTransportCaptureRecord:
    """Execute one live native planner transport qualification attempt."""
    messages = _follow_up_messages(
        available_evidence_references=available_evidence_references,
    )
    tools_schema = append_planner_action_context_schema([_probe_business_tool_schema()])
    response = adapter.generate_with_tools(
        messages,
        tools_schema,
        temperature=0.0,
        max_tokens=1024,
        tool_choice="auto",
    )
    reference_index = build_completed_observation_reference_index(messages)
    protocol_config = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ACTION_CONTEXT,
        available_evidence_references=tuple(reference_index),
        _reference_index_items=tuple(sorted(reference_index.items())),
    )
    try:
        action_context, business_calls = resolve_native_planner_protocol(
            response.tool_calls,
            protocol_config=protocol_config,
        )
    except Exception:
        return _capture_from_response(response, transport_success=False)
    basis_refs = action_context.evidence_basis_references if action_context is not None else ()
    purpose_present = bool(action_context is not None and action_context.purpose.strip())
    annotation_count = sum(
        1 for call in response.tool_calls if call.name == PLANNER_ACTION_CONTEXT_TOOL_ID
    )
    success = (
        action_context is not None
        and annotation_count == 1
        and len(business_calls) >= 1
    )
    return _capture_from_response(
        response,
        transport_success=success,
        basis_refs=basis_refs,
        purpose_present=purpose_present,
    )


def qualify_planner_transport(
    adapter: LLMAdapter,
    *,
    provider: str,
    required_attempts: int = _REQUIRED_QWEN_ATTEMPTS,
) -> PlannerTransportQualificationResult:
    """Run repeated transport attempts and evaluate the DS-E2E-12 gate."""
    captures: list[PlannerTransportCaptureRecord] = []
    successes = 0
    for _ in range(required_attempts):
        capture = run_one_planner_transport_attempt(adapter)
        captures.append(capture)
        if capture.transport_success:
            successes += 1
    return PlannerTransportQualificationResult(
        provider=provider,
        model=captures[-1].model if captures else None,
        required_attempts=required_attempts,
        successful_attempts=successes,
        captures=tuple(captures),
        gate_passed=successes >= required_attempts,
    )
