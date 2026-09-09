# © Artur Czarnecki. All rights reserved.

"""Provider-neutral strict tool call enforcement at the adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    CanonicalFunctionToolDefinition,
    CanonicalFunctionToolWireSchema,
    coerce_canonical_tool_definitions,
)
from intergrax.llm_adapters.contracts.strict_tool_call_validation import (
    StrictToolContractValidationError,
    validate_tool_calls_against_canonical_definitions,
)
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall


def _tool_name(definition: CanonicalFunctionToolDefinition) -> str:
    function = definition.wire_schema.get("function")
    if not isinstance(function, Mapping):
        raise StrictToolContractValidationError("canonical tool missing function object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise StrictToolContractValidationError("canonical tool missing function.name")
    return name


def resolve_canonical_tool_definitions(
    tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, object]],
) -> tuple[CanonicalFunctionToolDefinition, ...]:
    """Normalize request tools and reject duplicate canonical names before dispatch."""
    definitions = coerce_canonical_tool_definitions(tools)
    seen: dict[str, int] = {}
    for index, definition in enumerate(definitions):
        name = _tool_name(definition)
        if name in seen:
            raise StrictToolContractValidationError(
                f"duplicate canonical tool name {name!r} at tools[{index}] "
                f"(first declared at tools[{seen[name]}])"
            )
        seen[name] = index
    return definitions


def wire_schemas_from_definitions(
    definitions: Sequence[CanonicalFunctionToolDefinition],
) -> list[CanonicalFunctionToolWireSchema]:
    """Project canonical bindings to provider wire schemas (single schema source)."""
    return [definition.wire_schema for definition in definitions]


def enforce_strict_tool_call_conformance(
    tool_calls: Sequence[LLMToolCall],
    tool_definitions: Sequence[CanonicalFunctionToolDefinition | Mapping[str, object]],
) -> None:
    """Reject invalid provider tool calls before they cross the adapter boundary."""
    if not tool_calls:
        return
    validate_tool_calls_against_canonical_definitions(tool_calls, tool_definitions)
