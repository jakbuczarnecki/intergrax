# © Artur Czarnecki. All rights reserved.

"""Provider-neutral strict tool argument conformance contract (DS-E2E-13B)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

if TYPE_CHECKING:
    from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class CanonicalFunctionToolFunctionSchema(TypedDict):
    """Provider-neutral OpenAI-style function tool payload."""

    name: str
    description: NotRequired[str]
    parameters: NotRequired[dict[str, object]]


class CanonicalFunctionToolWireSchema(TypedDict):
    """Provider-neutral function-tool wire schema shared across adapters."""

    type: str
    function: CanonicalFunctionToolFunctionSchema


class ToolArgumentConformance(Enum):
    """Provider dispatch requirement for tool argument schema enforcement."""

    DEFAULT = "default"
    STRICT = "strict"


class StrictWireProjectionKind(Enum):
    """Provider-neutral strict-wire projection semantics for dispatch metadata."""

    ATOMIC_PLANNER_DISCRIMINATED_ACTIONS = "atomic_planner_discriminated_actions"


@dataclass(frozen=True, slots=True)
class ToolDispatchRequirements:
    """Typed platform dispatch metadata — not part of provider wire schema."""

    argument_conformance: ToolArgumentConformance = ToolArgumentConformance.DEFAULT
    strict_wire_projection: StrictWireProjectionKind | None = None

    @property
    def requires_strict_argument_conformance(self) -> bool:
        return self.argument_conformance == ToolArgumentConformance.STRICT


@dataclass(frozen=True, slots=True)
class CanonicalFunctionToolDefinition:
    """Canonical tool binding: wire schema plus typed dispatch metadata."""

    wire_schema: CanonicalFunctionToolWireSchema
    dispatch_requirements: ToolDispatchRequirements
    argument_guidance_text: str | None = None

    @property
    def requires_strict_argument_conformance(self) -> bool:
        return self.dispatch_requirements.requires_strict_argument_conformance


class StrictToolArgumentConformanceError(RuntimeError):
    """Tool schema requires provider-enforced argument conformance that the adapter lacks."""


_CANONICAL_BINDING_KEYS = frozenset(
    {"wire_schema", "dispatch_requirements", "argument_guidance_text"}
)


def _coerce_wire_schema(
    tool: Mapping[str, Any],
    index: int,
) -> CanonicalFunctionToolWireSchema:
    if tool.get("type") != "function":
        raise ValueError(
            f"tools[{index}] must be a function tool wire schema, got type={tool.get('type')!r}"
        )
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ValueError(f"tools[{index}] function tool missing nested 'function' object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"tools[{index}] function tool missing canonical name")
    function_schema: CanonicalFunctionToolFunctionSchema = {"name": name}
    description = function.get("description")
    if isinstance(description, str):
        function_schema["description"] = description
    parameters = function.get("parameters")
    if isinstance(parameters, dict):
        function_schema["parameters"] = parameters
    wire_schema: CanonicalFunctionToolWireSchema = {
        "type": "function",
        "function": function_schema,
    }
    return wire_schema


def coerce_canonical_tool_definitions(
    tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
) -> tuple[CanonicalFunctionToolDefinition, ...]:
    """Normalize request tools into immutable canonical bindings."""
    definitions: list[CanonicalFunctionToolDefinition] = []
    for index, tool in enumerate(tools):
        if isinstance(tool, CanonicalFunctionToolDefinition):
            definitions.append(tool)
            continue
        if not isinstance(tool, Mapping):
            raise TypeError(
                f"tools[{index}] must be CanonicalFunctionToolDefinition or wire schema mapping, "
                f"got {type(tool).__name__}"
            )
        binding_keys = _CANONICAL_BINDING_KEYS.intersection(tool.keys())
        if binding_keys:
            raise ValueError(
                f"tools[{index}] looks like a partial canonical binding "
                f"({sorted(binding_keys)}); pass CanonicalFunctionToolDefinition "
                "or a raw wire schema"
            )
        definitions.append(
            CanonicalFunctionToolDefinition(
                wire_schema=_coerce_wire_schema(tool, index),
                dispatch_requirements=ToolDispatchRequirements(),
                argument_guidance_text=None,
            )
        )
    return tuple(definitions)


def tool_definitions_require_strict_argument_conformance(
    definitions: Sequence[CanonicalFunctionToolDefinition],
) -> bool:
    """Return whether any canonical binding requires strict argument conformance."""
    return any(definition.requires_strict_argument_conformance for definition in definitions)


def assert_strict_tool_argument_conformance_supported(
    adapter: "LLMAdapter",
    tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
) -> None:
    """Fail closed when strict tools are dispatched to an adapter without capability."""
    definitions = coerce_canonical_tool_definitions(tools)
    if not tool_definitions_require_strict_argument_conformance(definitions):
        return
    if adapter.supports_strict_tool_argument_conformance():
        return
    raise StrictToolArgumentConformanceError(
        "tools schema requires provider-enforced strict argument conformance but "
        f"{adapter.__class__.__name__} does not support it"
    )
