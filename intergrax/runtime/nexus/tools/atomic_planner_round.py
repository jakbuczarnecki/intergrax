# © Artur Czarnecki. All rights reserved.

"""Discriminated atomic planner round transport (DS-E2E-12, ENG-6 certified).

Reserved protocol id ``intergrax.planner.round`` is NOT a ToolContract, capability, or
executable tool — exactly one provider-native function call represents one planner round
with optional action context and one or more typed business actions.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.llm_adapters.contracts.tool_call import (
    LLMToolCall,
    finalize_accepted_tool_call_identities,
)
from intergrax.runtime.nexus.tools.native_planner_action_context import (
    NativePlannerActionContext,
    NativePlannerActionContextError,
    NativePlannerProtocolConfig,
    parse_optional_planner_action_context_payload,
    validate_typed_planner_action_context,
)
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan
from intergrax.tools.exporters.openai import compute_openai_tools_schema_hash
from intergrax.tools.registry.runtime import ToolRegistry

PLANNER_ROUND_TOOL_ID = "intergrax.planner.round"


class AtomicPlannerRoundError(ValueError):
    """Invalid discriminated atomic planner round transport."""


@dataclass(frozen=True, slots=True)
class AtomicPlannerAction:
    tool_id: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class AtomicPlannerRoundDecision:
    action_context: NativePlannerActionContext | None
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


class _DiscriminatedActionInput(BaseModel):
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

    action_context: dict[str, object] | None = None
    actions: list[_DiscriminatedActionInput] = Field(default_factory=list)


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
    return tuple(sorted(entries, key=lambda entry: entry.tool_id))


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


def _build_discriminated_actions_schema(
    entries: Sequence[_BusinessToolSchemaEntry],
) -> dict[str, object]:
    if not entries:
        raise AtomicPlannerRoundError(
            "discriminated atomic round requires at least one admitted business tool"
        )
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
) -> dict[str, object]:
    """Derive discriminated atomic round parameters from admitted business schemas."""
    entries = extract_business_tool_schema_entries(business_schemas)
    actions_schema = _build_discriminated_actions_schema(entries)
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
) -> _OpenAIToolSchema:
    """Provider-neutral schema: single reserved ``intergrax.planner.round`` function."""
    parameters = build_atomic_planner_round_parameters_schema(business_schemas)
    return {
        "type": "function",
        "function": {
            "name": PLANNER_ROUND_TOOL_ID,
            "description": (
                "Declare one planner round: optional evidence basis and purpose, "
                "plus one or more business tool actions to execute. Planning transport only — "
                "not an executable business tool."
            ),
            "parameters": parameters,
        },
    }


def compute_atomic_planner_round_schema_hash(
    round_schema: Mapping[str, object],
) -> str:
    """Fingerprint for the derived atomic wrapper (does not replace business hash)."""
    return compute_openai_tools_schema_hash([round_schema])


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
    action_context_payload = validated.action_context
    if action_context_payload is None:
        action_context = None
    else:
        try:
            action_context = parse_optional_planner_action_context_payload(
                action_context_payload
            )
        except NativePlannerActionContextError as exc:
            raise AtomicPlannerRoundError(str(exc)) from exc
    return AtomicPlannerRoundDecision(
        action_context=action_context,
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
    if not decision.actions:
        raise AtomicPlannerRoundError(
            "atomic planner round with empty actions is not executable"
        )
    reference_index = protocol_config.reference_index()
    available = frozenset(protocol_config.available_evidence_references)
    if protocol_config.action_context_required:
        if decision.action_context is None:
            raise AtomicPlannerRoundError(
                "follow-up tool round requires action_context in atomic planner round"
            )
    if decision.action_context is None:
        return
    validate_typed_planner_action_context(
        decision.action_context,
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
        raise AtomicPlannerRoundError(
            "atomic mode requires exactly one intergrax.planner.round call"
        )
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


def mint_materialized_tool_calls_from_plan(
    tool_plan: ToolCallPlan,
) -> tuple[LLMToolCall, ...]:
    """Assign canonical accepted tool-call identities for materialized business actions."""
    provisional: list[LLMToolCall] = []
    for planned_call in tool_plan.calls:
        provisional.append(
            LLMToolCall(
                id="",
                name=planned_call.tool_id,
                arguments_json=json.dumps(
                    planned_call.input.model_dump(),
                    ensure_ascii=False,
                ),
            )
        )
    return finalize_accepted_tool_call_identities(provisional)


def atomic_round_schema_byte_size(schema: Mapping[str, object]) -> int:
    """Deterministic serialized schema size diagnostic."""
    return len(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
