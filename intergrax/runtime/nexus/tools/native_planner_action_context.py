# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — experimental typed planner action-context transport (planning layer)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall, validate_tool_call_identities
from intergrax.tools.exporters.schema import pydantic_parameters_schema

PLANNER_ACTION_CONTEXT_TOOL_ID = "intergrax.planner.action_context"


class NativePlannerActionContextError(ValueError):
    """Invalid typed planner action-context transport (DS-E2E-12)."""


class _PlannerActionContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_basis_references: list[str] = Field(default_factory=list)
    purpose: str

    @field_validator("evidence_basis_references", mode="before")
    @classmethod
    def _coerce_basis_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("evidence_basis_references must be an array of strings")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("evidence_basis_references must contain only strings")
            normalized.append(item)
        return normalized

    @field_validator("purpose")
    @classmethod
    def _non_empty_purpose(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("purpose must be non-empty")
        return stripped


@dataclass(frozen=True, slots=True)
class NativePlannerActionContext:
    """Model-authored ENG-6 justification transported as a native planner annotation."""

    evidence_basis_references: tuple[str, ...]
    purpose: str


@dataclass(frozen=True, slots=True)
class SplitNativePlannerToolCalls:
    """Deterministic partition of one native LLM tool-call batch."""

    annotation_calls: tuple[LLMToolCall, ...]
    business_tool_calls: tuple[LLMToolCall, ...]


@dataclass(frozen=True, slots=True)
class NativePlannerRoundTransport:
    """One atomic native planner response after protocol/business separation."""

    response: LLMAdapterResponse
    action_context: NativePlannerActionContext | None
    business_tool_calls: tuple[LLMToolCall, ...]
    annotation_call_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProcessNativePlannerTransportResult:
    """PoC outcome for one native planner round."""

    transport: NativePlannerRoundTransport
    is_executable_investigation_round: bool


class _OpenAIFunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


class _OpenAIToolSchema(TypedDict):
    type: str
    function: _OpenAIFunctionSchema


def planner_action_context_openai_tool_schema() -> _OpenAIToolSchema:
    """Model-facing schema for the reserved planner protocol annotation."""
    return {
        "type": "function",
        "function": {
            "name": PLANNER_ACTION_CONTEXT_TOOL_ID,
            "description": (
                "Declare the evidence basis and public purpose for the business tool "
                "actions selected in this same response. Required when prior evidence "
                "exists and follow-up business tools are requested."
            ),
            "parameters": pydantic_parameters_schema(_PlannerActionContextInput),
        },
    }


def append_planner_action_context_schema(
    business_schemas: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Append the reserved planner annotation schema to provider-facing business schemas."""
    materialized = [dict(entry) for entry in business_schemas]
    materialized.append(dict(planner_action_context_openai_tool_schema()))
    return materialized


def split_native_planner_tool_calls(
    tool_calls: Sequence[LLMToolCall],
) -> SplitNativePlannerToolCalls:
    """Separate reserved planner annotation calls from executable business tool calls."""
    annotation_calls: list[LLMToolCall] = []
    business_calls: list[LLMToolCall] = []
    for call in tool_calls:
        if call.name == PLANNER_ACTION_CONTEXT_TOOL_ID:
            annotation_calls.append(call)
            continue
        business_calls.append(call)
    return SplitNativePlannerToolCalls(
        annotation_calls=tuple(annotation_calls),
        business_tool_calls=tuple(business_calls),
    )


def _parse_basis_references(raw_references: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for reference in raw_references:
        stripped = reference.strip()
        if not stripped:
            raise NativePlannerActionContextError(
                "planner action context contains empty evidence_basis_references entry"
            )
        if stripped in seen:
            raise NativePlannerActionContextError(
                f"duplicate basis evidence reference: {stripped}"
            )
        seen.add(stripped)
        ordered.append(stripped)
    return tuple(ordered)


def parse_planner_action_context_call(call: LLMToolCall) -> NativePlannerActionContext:
    """Parse and validate one reserved planner annotation tool call."""
    if call.name != PLANNER_ACTION_CONTEXT_TOOL_ID:
        raise NativePlannerActionContextError(
            f"unexpected planner annotation tool id: {call.name!r}"
        )
    try:
        payload = json.loads(call.arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise NativePlannerActionContextError(
            "planner action context arguments JSON is malformed"
        ) from exc
    if not isinstance(payload, dict):
        raise NativePlannerActionContextError(
            "planner action context arguments must be a JSON object"
        )
    try:
        validated = _PlannerActionContextInput.model_validate(payload)
    except Exception as exc:
        raise NativePlannerActionContextError(
            f"planner action context schema validation failed: {exc}"
        ) from exc
    return NativePlannerActionContext(
        evidence_basis_references=_parse_basis_references(
            validated.evidence_basis_references
        ),
        purpose=validated.purpose,
    )


def validate_typed_planner_action_context(
    context: NativePlannerActionContext,
    *,
    available_evidence_references: frozenset[str],
    reference_index: dict[str, str],
) -> None:
    """Apply canonical ENG-6 basis semantics to a typed planner annotation."""
    if available_evidence_references and not context.evidence_basis_references:
        raise NativePlannerActionContextError(
            "follow-up tool round requires explicit evidence basis "
            f"(available_evidence_count={len(available_evidence_references)}, "
            "basis_count=0)"
        )
    if not context.evidence_basis_references:
        return
    for reference in context.evidence_basis_references:
        if reference not in reference_index:
            raise NativePlannerActionContextError(
                f"unknown basis evidence reference: {reference}"
            )


def _resolve_action_context(
    annotation_calls: Sequence[LLMToolCall],
    *,
    available_evidence_references: frozenset[str],
    reference_index: dict[str, str],
) -> NativePlannerActionContext | None:
    if not annotation_calls:
        return None
    if len(annotation_calls) > 1:
        raise NativePlannerActionContextError(
            "planner action context cardinality violation: expected at most one annotation"
        )
    context = parse_planner_action_context_call(annotation_calls[0])
    validate_typed_planner_action_context(
        context,
        available_evidence_references=available_evidence_references,
        reference_index=reference_index,
    )
    return context


def process_native_planner_tool_response(
    response: LLMAdapterResponse,
    *,
    available_evidence_references: frozenset[str],
    reference_index: dict[str, str],
) -> ProcessNativePlannerTransportResult:
    """Split, validate, and classify one native planner tool response (PoC seam)."""
    validate_tool_call_identities(response.tool_calls)
    split = split_native_planner_tool_calls(response.tool_calls)
    business_calls = split.business_tool_calls
    annotation_ids = tuple(call.id for call in split.annotation_calls)

    if split.annotation_calls and not business_calls:
        return ProcessNativePlannerTransportResult(
            transport=NativePlannerRoundTransport(
                response=response,
                action_context=_resolve_action_context(
                    split.annotation_calls,
                    available_evidence_references=available_evidence_references,
                    reference_index=reference_index,
                ),
                business_tool_calls=(),
                annotation_call_ids=annotation_ids,
            ),
            is_executable_investigation_round=False,
        )

    if available_evidence_references and business_calls and not split.annotation_calls:
        raise NativePlannerActionContextError(
            "follow-up tool round requires exactly one planner action context annotation"
        )

    action_context = _resolve_action_context(
        split.annotation_calls,
        available_evidence_references=available_evidence_references,
        reference_index=reference_index,
    )
    return ProcessNativePlannerTransportResult(
        transport=NativePlannerRoundTransport(
            response=response,
            action_context=action_context,
            business_tool_calls=business_calls,
            annotation_call_ids=annotation_ids,
        ),
        is_executable_investigation_round=bool(business_calls),
    )
