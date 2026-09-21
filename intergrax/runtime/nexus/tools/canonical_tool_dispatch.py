# © Artur Czarnecki. All rights reserved.

"""Runtime ownership: wire tool schemas → canonical LLM adapter dispatch bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from intergrax.knowledge.contracts.validation import JsonValue, validate_json_value
from intergrax.llm_adapters._shared.strict_tool_enforcement import (
    resolve_canonical_tool_definitions,
)
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    CanonicalFunctionToolDefinition,
    coerce_canonical_tool_definitions,
)


def _validate_raw_wire_tool_schema(
    tool: Mapping[str, object],
    index: int,
) -> Mapping[str, JsonValue]:
    if not isinstance(tool, Mapping):
        raise TypeError(
            f"tools[{index}] must be CanonicalFunctionToolDefinition or wire schema mapping, "
            f"got {type(tool).__name__}"
        )
    try:
        validated = validate_json_value(tool, field_name=f"tools[{index}]")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(validated, dict):
        raise ValueError(
            f"tools[{index}] must be a JSON object, got {type(validated).__name__}"
        )
    return validated


def materialize_canonical_tool_definitions_for_llm_dispatch(
    tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, object]],
) -> tuple[CanonicalFunctionToolDefinition, ...]:
    """Materialize OpenAI-shaped wire schemas before the canonical LLM adapter boundary."""
    wire_tools: list[CanonicalFunctionToolDefinition | Mapping[str, JsonValue]] = []
    for index, tool in enumerate(tools):
        if isinstance(tool, CanonicalFunctionToolDefinition):
            wire_tools.append(tool)
            continue
        wire_tools.append(_validate_raw_wire_tool_schema(tool, index))
    definitions = coerce_canonical_tool_definitions(wire_tools)
    return resolve_canonical_tool_definitions(definitions)
