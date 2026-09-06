# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — live atomic planner round transport qualification (Variants A/B)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    PLANNER_ROUND_TOOL_ID,
    AtomicPlannerRoundSchemaVariant,
    build_atomic_planner_round_schema,
    extract_business_tool_schema_entries,
    materialize_atomic_round_to_tool_plan,
    planner_round_tool_choice_for_provider,
    resolve_atomic_planner_round_calls,
)
from intergrax.runtime.nexus.tools.investigation_proof import format_investigation_follow_up_context
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerProtocolConfig,
    NativePlannerProtocolMode,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.builder import tools_agent_make_contract

_LIVE_FLAG = "INTERGRAX_DS_E2E_12_LIVE"
_REQUIRED_QWEN_ATTEMPTS = 3


class AtomicTransportOutcome(Enum):
    SUCCESS = "success"
    SCHEMA_REJECTED = "schema_rejected"
    NO_ATOMIC_CALL = "no_atomic_call"
    PARSE_FAILED = "parse_failed"
    VALIDATION_FAILED = "validation_failed"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True, slots=True)
class AtomicTransportCaptureRecord:
    variant: str
    model: str | None
    provider: str | None
    outcome: str
    atomic_call_present: bool
    action_count: int
    basis_refs: tuple[str, ...]
    purpose_present: bool
    runtime_valid: bool
    provider_error: str | None
    response_id: str | None


@dataclass(frozen=True, slots=True)
class AtomicTransportQualificationResult:
    provider: str
    variant: AtomicPlannerRoundSchemaVariant
    model: str | None
    required_attempts: int
    successful_attempts: int
    captures: tuple[AtomicTransportCaptureRecord, ...]
    gate_passed: bool


class _TelemetryIn(BaseModel):
    pass


class _AttendanceIn(BaseModel):
    line_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class _AttendanceOut(BaseModel):
    line_id: str
    window: str
    present: int


class _FilterClause(BaseModel):
    field: str
    operator: str
    value: str


class _MetricsIn(BaseModel):
    metric_name: str = Field(min_length=1)
    filters: list[_FilterClause] = Field(default_factory=list)


class _MetricsOut(BaseModel):
    metric_name: str
    value: float


class _NoopHandler:
    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        _ = request
        raise NotImplementedError("PoC transport harness does not execute tools")


def live_atomic_transport_enabled() -> bool:
    return os.environ.get(_LIVE_FLAG, "").strip() == "1"


def poc_business_tool_schemas() -> list[dict[str, object]]:
    """Three materially different admitted business schemas for PoC."""
    return [
        {
            "type": "function",
            "function": {
                "name": "production.telemetry.read",
                "description": "Read production telemetry snapshot (zero/low args).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "production.staffing.attendance.read",
                "description": "Read staffing attendance for a line and window.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_id": {"type": "string"},
                        "window": {"type": "string"},
                    },
                    "required": ["line_id", "window"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "production.metrics.query",
                "description": "Query metrics with nested filter clauses.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "operator": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                                "required": ["field", "operator", "value"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["metric_name", "filters"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def poc_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        tools_agent_make_contract("production.telemetry.read", _TelemetryIn, BaseModel),
        _NoopHandler(),
    )
    registry.register(
        tools_agent_make_contract(
            "production.staffing.attendance.read",
            _AttendanceIn,
            _AttendanceOut,
        ),
        _NoopHandler(),
    )
    registry.register(
        tools_agent_make_contract("production.metrics.query", _MetricsIn, _MetricsOut),
        _NoopHandler(),
    )
    return registry


def _follow_up_messages(
    *,
    available_evidence_references: Sequence[str],
    require_multi_action: bool,
) -> list[ChatMessage]:
    multi_hint = (
        " Request two business actions in one atomic round: "
        "production.telemetry.read and production.staffing.attendance.read."
        if require_multi_action
        else ""
    )
    return [
        ChatMessage(
            role="system",
            content=(
                "You are an incident investigator. Emit exactly one "
                f"{PLANNER_ROUND_TOOL_ID} call containing action_context and business actions."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Investigate elevated error rate for checkout service. "
                f"Prior evidence reference available: {available_evidence_references[0]}."
            ),
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
                "Continue investigation using one atomic planner round. Include action_context "
                f"with basis reference {available_evidence_references[0]} and a non-empty purpose, "
                "plus at least one admitted business action."
                f"{multi_hint}"
            ),
        ),
    ]


def _capture_from_response(
    response: LLMAdapterResponse | None,
    *,
    variant: AtomicPlannerRoundSchemaVariant,
    outcome: AtomicTransportOutcome,
    action_count: int = 0,
    basis_refs: tuple[str, ...] = (),
    purpose_present: bool = False,
    runtime_valid: bool = False,
    provider_error: str | None = None,
) -> AtomicTransportCaptureRecord:
    atomic_present = False
    if response is not None:
        atomic_present = any(call.name == PLANNER_ROUND_TOOL_ID for call in response.tool_calls)
    return AtomicTransportCaptureRecord(
        variant=variant.value,
        model=response.model if response is not None else None,
        provider=response.provider if response is not None else None,
        outcome=outcome.value,
        atomic_call_present=atomic_present,
        action_count=action_count,
        basis_refs=basis_refs,
        purpose_present=purpose_present,
        runtime_valid=runtime_valid,
        provider_error=provider_error,
        response_id=response.response_id if response is not None else None,
    )


def run_one_atomic_transport_attempt(
    adapter: LLMAdapter,
    *,
    variant: AtomicPlannerRoundSchemaVariant,
    provider: str,
    available_evidence_references: Sequence[str] = ("observation.production.telemetry.read.prior",),
    require_multi_action: bool = False,
) -> AtomicTransportCaptureRecord:
    messages = _follow_up_messages(
        available_evidence_references=available_evidence_references,
        require_multi_action=require_multi_action,
    )
    business_schemas = poc_business_tool_schemas()
    round_schema = build_atomic_planner_round_schema(business_schemas, variant=variant)
    tool_choice = planner_round_tool_choice_for_provider(provider)
    try:
        response = adapter.generate_with_tools(
            messages,
            [round_schema],
            temperature=0.0,
            max_tokens=2048,
            tool_choice=tool_choice,
        )
    except Exception as exc:
        message = str(exc)
        if "schema" in message.lower() or "tool" in message.lower():
            return _capture_from_response(
                None,
                variant=variant,
                outcome=AtomicTransportOutcome.SCHEMA_REJECTED,
                provider_error=message,
            )
        return _capture_from_response(
            None,
            variant=variant,
            outcome=AtomicTransportOutcome.PROVIDER_ERROR,
            provider_error=message,
        )

    if not any(call.name == PLANNER_ROUND_TOOL_ID for call in response.tool_calls):
        return _capture_from_response(
            response,
            variant=variant,
            outcome=AtomicTransportOutcome.NO_ATOMIC_CALL,
        )

    reference_index = {
        reference: reference for reference in available_evidence_references
    }
    protocol_config = NativePlannerProtocolConfig(
        mode=NativePlannerProtocolMode.INVESTIGATION_ACTION_CONTEXT,
        available_evidence_references=tuple(reference_index),
        _reference_index_items=tuple(sorted(reference_index.items())),
    )
    admitted = frozenset(
        entry.tool_id for entry in extract_business_tool_schema_entries(business_schemas)
    )
    try:
        decision = resolve_atomic_planner_round_calls(
            response.tool_calls,
            protocol_config=protocol_config,
            admitted_tool_ids=admitted,
        )
        registry = poc_tool_registry()
        materialize_atomic_round_to_tool_plan(decision, registry)
    except Exception as exc:
        return _capture_from_response(
            response,
            variant=variant,
            outcome=AtomicTransportOutcome.VALIDATION_FAILED,
            provider_error=str(exc),
            action_count=len(response.tool_calls),
        )

    basis_refs = (
        decision.action_context.evidence_basis_references
        if decision.action_context is not None
        else ()
    )
    purpose_present = bool(decision.action_context is not None and decision.action_context.purpose)
    success = (
        decision.action_context is not None
        and len(basis_refs) >= 1
        and purpose_present
        and len(decision.actions) >= 1
    )
    return _capture_from_response(
        response,
        variant=variant,
        outcome=AtomicTransportOutcome.SUCCESS if success else AtomicTransportOutcome.PARSE_FAILED,
        action_count=len(decision.actions),
        basis_refs=basis_refs,
        purpose_present=purpose_present,
        runtime_valid=True,
    )


def qualify_atomic_planner_transport(
    adapter: LLMAdapter,
    *,
    provider: str,
    variant: AtomicPlannerRoundSchemaVariant,
    required_attempts: int = _REQUIRED_QWEN_ATTEMPTS,
    require_multi_action: bool = False,
) -> AtomicTransportQualificationResult:
    captures: list[AtomicTransportCaptureRecord] = []
    successes = 0
    for _ in range(required_attempts):
        capture = run_one_atomic_transport_attempt(
            adapter,
            variant=variant,
            provider=provider,
            require_multi_action=require_multi_action,
        )
        captures.append(capture)
        if capture.outcome == AtomicTransportOutcome.SUCCESS.value and capture.runtime_valid:
            successes += 1
    return AtomicTransportQualificationResult(
        provider=provider,
        variant=variant,
        model=captures[-1].model if captures else None,
        required_attempts=required_attempts,
        successful_attempts=successes,
        captures=tuple(captures),
        gate_passed=successes >= required_attempts,
    )
