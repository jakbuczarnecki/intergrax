# © Artur Czarnecki. All rights reserved.

"""Typed native planner action-context transport (DS-E2E-12, Tool Planning ownership)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.exporters.schema import pydantic_parameters_schema

# Reserved planner protocol record — NOT a ToolContract, capability, or executable Tool.
PLANNER_ACTION_CONTEXT_TOOL_ID = "intergrax.planner.action_context"


class NativePlannerActionContextError(ValueError):
    """Invalid typed planner action-context transport (DS-E2E-12)."""


class NativePlannerProtocolMode(Enum):
    """Explicit native planner protocol transport mode."""

    NONE = "none"
    INVESTIGATION_ACTION_CONTEXT = "investigation_action_context"


@dataclass(frozen=True, slots=True)
class NativePlannerProtocolConfig:
    """Whether annotation transport is active and which evidence refs are admissible."""

    mode: NativePlannerProtocolMode = NativePlannerProtocolMode.NONE
    available_evidence_references: tuple[str, ...] = ()
    _reference_index_items: tuple[tuple[str, str], ...] = ()

    @property
    def protocol_active(self) -> bool:
        return self.mode == NativePlannerProtocolMode.INVESTIGATION_ACTION_CONTEXT

    @property
    def action_context_required(self) -> bool:
        return self.protocol_active and bool(self.available_evidence_references)

    def reference_index(self) -> dict[str, str]:
        return dict(self._reference_index_items)


NATIVE_PLANNER_PROTOCOL_NONE = NativePlannerProtocolConfig()


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
    """Model-authored evidence basis and public purpose for one planner round."""

    evidence_basis_references: tuple[str, ...]
    purpose: str


@dataclass(frozen=True, slots=True)
class SplitNativePlannerToolCalls:
    """Deterministic partition of one native LLM tool-call batch."""

    annotation_calls: tuple[LLMToolCall, ...]
    business_tool_calls: tuple[LLMToolCall, ...]


@dataclass(frozen=True, slots=True)
class NativePlannerRound:
    """One atomic native planner response after protocol/business separation."""

    response: LLMAdapterResponse
    business_tool_calls: tuple[LLMToolCall, ...]
    tool_plan: ToolCallPlan
    action_context: NativePlannerActionContext | None


class _OpenAIFunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


class _OpenAIToolSchema(TypedDict):
    type: str
    function: _OpenAIFunctionSchema


def build_native_planner_action_context_schema() -> _OpenAIToolSchema:
    """Model-facing schema for the reserved planner protocol annotation."""
    return {
        "type": "function",
        "function": {
            "name": PLANNER_ACTION_CONTEXT_TOOL_ID,
            "description": (
                "Declare the evidence basis and public purpose for the business tool "
                "actions selected in this same response. Required when prior evidence "
                "exists and follow-up business tools are requested. Planning metadata "
                "only — not an executable tool."
            ),
            "parameters": pydantic_parameters_schema(_PlannerActionContextInput),
        },
    }


def append_planner_action_context_schema(
    business_schemas: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Append reserved planner annotation schema after validated business schemas."""
    materialized = [dict(entry) for entry in business_schemas]
    materialized.append(dict(build_native_planner_action_context_schema()))
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


def resolve_native_planner_protocol(
    tool_calls: Sequence[LLMToolCall],
    *,
    protocol_config: NativePlannerProtocolConfig,
) -> tuple[NativePlannerActionContext | None, tuple[LLMToolCall, ...]]:
    """Split, validate, and classify one native planner tool response."""
    if not protocol_config.protocol_active:
        split = split_native_planner_tool_calls(tool_calls)
        if split.annotation_calls:
            raise NativePlannerActionContextError(
                "unexpected planner annotation when protocol transport is inactive"
            )
        return None, split.business_tool_calls

    split = split_native_planner_tool_calls(tool_calls)
    business_calls = split.business_tool_calls
    reference_index = protocol_config.reference_index()
    available = frozenset(protocol_config.available_evidence_references)

    if split.annotation_calls and not business_calls:
        raise NativePlannerActionContextError(
            "planner action context without business tool calls is not executable"
        )

    if protocol_config.action_context_required and business_calls and not split.annotation_calls:
        raise NativePlannerActionContextError(
            "follow-up tool round requires exactly one planner action context annotation"
        )

    action_context = _resolve_action_context(
        split.annotation_calls,
        available_evidence_references=available,
        reference_index=reference_index,
    )
    return action_context, business_calls
