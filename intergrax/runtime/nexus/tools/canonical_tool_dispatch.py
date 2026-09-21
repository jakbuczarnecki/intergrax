# © Artur Czarnecki. All rights reserved.

"""Runtime ownership: wire tool schemas → canonical LLM adapter dispatch bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from intergrax.llm_adapters._shared.strict_tool_enforcement import (
    resolve_canonical_tool_definitions,
)
from intergrax.llm_adapters.contracts.serialized_value import JsonValue
from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    CanonicalFunctionToolDefinition,
    coerce_canonical_tool_definitions,
)


def materialize_canonical_tool_definitions_for_llm_dispatch(
    tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, object]],
) -> tuple[CanonicalFunctionToolDefinition, ...]:
    """Materialize OpenAI-shaped wire schemas before the canonical LLM adapter boundary."""
    wire_tools = cast(
        "Sequence[CanonicalFunctionToolDefinition | Mapping[str, JsonValue]]",
        tools,
    )
    definitions = coerce_canonical_tool_definitions(wire_tools)
    return resolve_canonical_tool_definitions(definitions)
