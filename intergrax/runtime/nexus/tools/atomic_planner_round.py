# © Artur Czarnecki. All rights reserved.

"""Atomic planner round transport PoC (DS-E2E-12).

Architecture decision (PoC qualification):
Sibling ``intergrax.planner.action_context`` + business tool calls: capability proven in
isolated PoC, reliability rejected by full Qwen32 + GPT-4.1 controls. Do not invest in
prompt-only sibling-call fixes.

Reserved protocol id ``intergrax.planner.round`` is NOT a ToolContract, capability, or
executable tool — exactly one provider-native function call represents one planner round.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm_adapters.providers._openai_schema import project_json_schema_for_openai_strict
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerActionContext,
    NativePlannerActionContextError,
    NativePlannerProtocolConfig,
    validate_typed_planner_action_context,
)
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.exporters.openai import compute_openai_tools_schema_hash
from intergrax.tools.registry.runtime import ToolRegistry

PLANNER_ROUND_TOOL_ID = "intergrax.planner.round"


class AtomicPlannerRoundError(ValueError):
    """Invalid atomic planner round transport (DS-E2E-12 PoC)."""


class AtomicPlannerRoundSchemaVariant(Enum):
    """Model-facing atomic envelope schema shape."""

    GENERIC_ENVELOPE = "generic_envelope"
    DISCRIMINATED_ACTIONS = "discriminated_actions"


@dataclass(frozen=True, slots=True)
class AtomicPlannerActionContext:
    evidence_basis_references: tuple[str, ...]
    purpose: str


@dataclass(frozen=True, slots=True)
class AtomicPlannerAction:
    tool_id: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class AtomicPlannerRoundDecision:
    action_context: AtomicPlannerActionContext | None
    actions: tuple[AtomicPlannerAction, ...]


@dataclass(frozen=True, slots=True)
class _BusinessToolSchemaEntry:
    tool_id: str
    parameters: dict[str, object]


class _OpenAIFunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


class _OpenAIToolSchema(TypedDict):
    type: str
    function: _OpenAIFunctionSchema


class _ActionContextInput(BaseModel):
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


class _GenericActionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: str
    arguments: dict[str, object] = Field(default_factory=dict)

    @field_validator("tool_id")
    @classmethod
    def _non_empty_tool_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("tool_id must be non-empty")
        return stripped


class _AtomicRoundInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_context: _ActionContextInput | None = None
    actions: list[_GenericActionInput] = Field(default_factory=list)


def extract_business_tool_schema_entries(
    business_schemas: Sequence[Mapping[str, object]],
) -> tuple[_BusinessToolSchemaEntry, ...]:
    """Typed transformer: only ``function.name`` and ``function.parameters`` per tool."""
    entries: list[_BusinessToolSchemaEntry] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(business_schemas):
        if raw_entry.get("type") != "function":
            raise AtomicPlannerRoundError(
                f"business schema entry {index} is not a function tool"
            )
        function = raw_entry.get("function")
        if not isinstance(function, Mapping):
            raise AtomicPlannerRoundError(
                f"business schema entry {index} missing function object"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AtomicPlannerRoundError(
                f"business schema entry {index} missing function.name"
            )
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            raise AtomicPlannerRoundError(
                f"business schema entry {index} missing function.parameters"
            )
        if name in seen:
            raise AtomicPlannerRoundError(f"duplicate business tool id in schemas: {name}")
        seen.add(name)
        entries.append(
            _BusinessToolSchemaEntry(
                tool_id=name,
                parameters=copy.deepcopy(dict(parameters)),
            )
        )
    return tuple(entries)


def _action_context_property_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "evidence_basis_references": {
                "type": "array",
                "items": {"type": "string"},
            },
            "purpose": {"type": "string"},
        },
        "required": ["evidence_basis_references", "purpose"],
        "additionalProperties": False,
    }


def _build_variant_a_actions_schema(
    admitted_tool_ids: Sequence[str],
) -> dict[str, object]:
    if not admitted_tool_ids:
        raise AtomicPlannerRoundError("variant A requires at least one admitted business tool")
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "tool_id": {"type": "string", "enum": list(admitted_tool_ids)},
                "arguments": {"type": "object"},
            },
            "required": ["tool_id", "arguments"],
            "additionalProperties": False,
        },
    }


def _build_variant_b_actions_schema(
    entries: Sequence[_BusinessToolSchemaEntry],
) -> dict[str, object]:
    if not entries:
        raise AtomicPlannerRoundError("variant B requires at least one admitted business tool")
    one_of: list[dict[str, object]] = []
    for entry in entries:
        one_of.append(
            {
                "type": "object",
                "properties": {
                    "tool_id": {"const": entry.tool_id},
                    "arguments": copy.deepcopy(entry.parameters),
                },
                "required": ["tool_id", "arguments"],
                "additionalProperties": False,
            }
        )
    return {
        "type": "array",
        "items": {"oneOf": one_of},
    }


def build_atomic_planner_round_parameters_schema(
    business_schemas: Sequence[Mapping[str, object]],
    *,
    variant: AtomicPlannerRoundSchemaVariant,
) -> dict[str, object]:
    """Derive atomic round parameters from validated admitted business schemas."""
    entries = extract_business_tool_schema_entries(business_schemas)
    admitted_tool_ids = tuple(entry.tool_id for entry in entries)
    if variant == AtomicPlannerRoundSchemaVariant.GENERIC_ENVELOPE:
        actions_schema = _build_variant_a_actions_schema(admitted_tool_ids)
    else:
        actions_schema = _build_variant_b_actions_schema(entries)
    return {
        "type": "object",
        "properties": {
            "action_context": _action_context_property_schema(),
            "actions": actions_schema,
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def build_atomic_planner_round_schema(
    business_schemas: Sequence[Mapping[str, object]],
    *,
    variant: AtomicPlannerRoundSchemaVariant,
    strict_provider_schema: bool = True,
) -> _OpenAIToolSchema:
    """Model-facing schema: single reserved ``intergrax.planner.round`` function."""
    parameters = build_atomic_planner_round_parameters_schema(
        business_schemas,
        variant=variant,
    )
    if strict_provider_schema:
        parameters = project_json_schema_for_openai_strict(parameters)
    return {
        "type": "function",
        "function": {
            "name": PLANNER_ROUND_TOOL_ID,
            "description": (
                "Declare one atomic planner round: optional evidence basis and purpose, "
                "plus one or more business tool actions to execute. Planning transport only — "
                "not an executable business tool."
            ),
            "parameters": parameters,
        },
    }


def compute_atomic_planner_round_schema_hash(
    round_schema: Mapping[str, object],
) -> str:
    """PoC fingerprint for the derived atomic wrapper (does not replace business hash)."""
    return compute_openai_tools_schema_hash([round_schema])


def planner_round_tool_choice_for_provider(provider: str) -> str | dict[str, str]:
    """Forceable transport selection — provider-neutral where supported."""
    if provider == "ollama":
        return "required"
    return {"type": "function", "name": PLANNER_ROUND_TOOL_ID}


def _parse_basis_references(raw_references: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for reference in raw_references:
        stripped = reference.strip()
        if not stripped:
            raise AtomicPlannerRoundError(
                "action context contains empty evidence_basis_references entry"
            )
        if stripped in seen:
            raise AtomicPlannerRoundError(f"duplicate basis evidence reference: {stripped}")
        seen.add(stripped)
        ordered.append(stripped)
    return tuple(ordered)


def _materialize_action_context(
    payload: _ActionContextInput | None,
) -> AtomicPlannerActionContext | None:
    if payload is None:
        return None
    return AtomicPlannerActionContext(
        evidence_basis_references=_parse_basis_references(payload.evidence_basis_references),
        purpose=payload.purpose,
    )


def parse_atomic_planner_round_payload(payload: Mapping[str, object]) -> AtomicPlannerRoundDecision:
    """Parse one validated atomic round JSON object."""
    try:
        validated = _AtomicRoundInput.model_validate(dict(payload))
    except Exception as exc:
        raise AtomicPlannerRoundError(
            f"atomic planner round schema validation failed: {exc}"
        ) from exc
    actions: list[AtomicPlannerAction] = []
    for action in validated.actions:
        actions.append(
            AtomicPlannerAction(
                tool_id=action.tool_id,
                arguments_json=json.dumps(action.arguments, ensure_ascii=False),
            )
        )
    return AtomicPlannerRoundDecision(
        action_context=_materialize_action_context(validated.action_context),
        actions=tuple(actions),
    )


def parse_atomic_planner_round_call(call: LLMToolCall) -> AtomicPlannerRoundDecision:
    """Parse and validate one reserved atomic planner round tool call."""
    if call.name != PLANNER_ROUND_TOOL_ID:
        raise AtomicPlannerRoundError(f"unexpected atomic planner tool id: {call.name!r}")
    try:
        payload = json.loads(call.arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise AtomicPlannerRoundError("atomic planner round arguments JSON is malformed") from exc
    if not isinstance(payload, dict):
        raise AtomicPlannerRoundError("atomic planner round arguments must be a JSON object")
    return parse_atomic_planner_round_payload(payload)


def validate_atomic_action_context_requirement(
    decision: AtomicPlannerRoundDecision,
    *,
    protocol_config: NativePlannerProtocolConfig,
) -> None:
    """ENG-6 semantics for typed action context inside the atomic envelope."""
    if not protocol_config.protocol_active:
        return
    reference_index = protocol_config.reference_index()
    available = frozenset(protocol_config.available_evidence_references)
    if protocol_config.action_context_required and decision.actions:
        if decision.action_context is None:
            raise AtomicPlannerRoundError(
                "follow-up tool round requires action_context in atomic planner round"
            )
    if decision.action_context is None:
        return
    native_context = NativePlannerActionContext(
        evidence_basis_references=decision.action_context.evidence_basis_references,
        purpose=decision.action_context.purpose,
    )
    validate_typed_planner_action_context(
        native_context,
        available_evidence_references=available,
        reference_index=reference_index,
    )


def resolve_atomic_planner_round_calls(
    tool_calls: Sequence[LLMToolCall],
    *,
    protocol_config: NativePlannerProtocolConfig,
    admitted_tool_ids: frozenset[str],
) -> AtomicPlannerRoundDecision:
    """Cardinality gate: exactly one ``intergrax.planner.round``, no sibling business calls."""
    round_calls: list[LLMToolCall] = []
    other_calls: list[LLMToolCall] = []
    for call in tool_calls:
        if call.name == PLANNER_ROUND_TOOL_ID:
            round_calls.append(call)
            continue
        other_calls.append(call)
    if other_calls:
        names = ", ".join(sorted({call.name for call in other_calls}))
        raise AtomicPlannerRoundError(
            f"atomic mode rejects sibling business provider calls: {names}"
        )
    if not round_calls:
        raise AtomicPlannerRoundError("atomic mode requires exactly one intergrax.planner.round call")
    if len(round_calls) > 1:
        raise AtomicPlannerRoundError(
            "atomic mode cardinality violation: expected exactly one intergrax.planner.round"
        )
    decision = parse_atomic_planner_round_call(round_calls[0])
    validate_atomic_action_context_requirement(
        decision,
        protocol_config=protocol_config,
    )
    for action in decision.actions:
        if action.tool_id not in admitted_tool_ids:
            raise AtomicPlannerRoundError(f"unknown admitted business tool id: {action.tool_id}")
    return decision


def materialize_atomic_round_to_tool_plan(
    decision: AtomicPlannerRoundDecision,
    registry: ToolRegistry,
    *,
    allowed_tool_ids: frozenset[str] | None = None,
    step_id: str = "tool",
) -> ToolCallPlan:
    """Materialize model-declared actions through ToolRegistry → PlannedToolCall → ToolCallPlan."""
    calls: list[PlannedToolCall] = []
    for action in decision.actions:
        if allowed_tool_ids is not None and action.tool_id not in allowed_tool_ids:
            raise AtomicPlannerRoundError(f"tool id not allowed for this round: {action.tool_id}")
        if not registry.has(action.tool_id):
            raise AtomicPlannerRoundError(f"tool not registered: {action.tool_id}")
        try:
            args = json.loads(action.arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise AtomicPlannerRoundError(
                f"business action arguments JSON is malformed for {action.tool_id}"
            ) from exc
        if not isinstance(args, dict):
            raise AtomicPlannerRoundError(
                f"business action arguments must be a JSON object for {action.tool_id}"
            )
        registered = registry.get(action.tool_id)
        validated = registered.contract.input_schema.model_validate(args)
        calls.append(
            PlannedToolCall(
                step_id=step_id,
                tool_id=action.tool_id,
                input=validated,
            )
        )
    return ToolCallPlan(calls=calls)


def atomic_round_schema_byte_size(schema: Mapping[str, object]) -> int:
    """Approximate serialized schema size for PoC scorecard."""
    return len(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
