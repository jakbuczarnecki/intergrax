# © Artur Czarnecki. All rights reserved.

"""Provider-neutral strict tool argument conformance contract (DS-E2E-13B)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

REQUIRES_STRICT_ARGUMENT_CONFORMANCE_FIELD = "requires_strict_argument_conformance"


class StrictToolArgumentConformanceError(RuntimeError):
    """Tool schema requires provider-enforced argument conformance that the adapter lacks."""


def function_tool_requires_strict_argument_conformance(
    tool: Mapping[str, Any],
) -> bool:
    """Return whether a canonical function tool requires provider-side strict conformance."""
    if tool.get("type") != "function":
        return False
    fn = tool.get("function")
    if not isinstance(fn, Mapping):
        return False
    return fn.get(REQUIRES_STRICT_ARGUMENT_CONFORMANCE_FIELD) is True


def tools_schema_requires_strict_argument_conformance(
    tools_schema: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether any tool in the schema list requires strict argument conformance."""
    return any(function_tool_requires_strict_argument_conformance(tool) for tool in tools_schema)


def assert_strict_tool_argument_conformance_supported(
    adapter: LLMAdapter,
    tools_schema: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed when strict tools are dispatched to an adapter without capability."""
    if not tools_schema_requires_strict_argument_conformance(tools_schema):
        return
    if adapter.supports_strict_tool_argument_conformance():
        return
    raise StrictToolArgumentConformanceError(
        "tools schema requires provider-enforced strict argument conformance but "
        f"{adapter.__class__.__name__} does not support it"
    )
