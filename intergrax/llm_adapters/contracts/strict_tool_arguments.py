# © Artur Czarnecki. All rights reserved.

"""Provider-neutral strict tool argument conformance contract (DS-E2E-13B)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class ToolArgumentConformance(Enum):
    """Provider dispatch requirement for tool argument schema enforcement."""

    DEFAULT = "default"
    STRICT = "strict"


@dataclass(frozen=True, slots=True)
class ToolDispatchRequirements:
    """Typed platform dispatch metadata — not part of provider wire schema."""

    argument_conformance: ToolArgumentConformance = ToolArgumentConformance.DEFAULT

    @property
    def requires_strict_argument_conformance(self) -> bool:
        return self.argument_conformance == ToolArgumentConformance.STRICT


@dataclass(frozen=True, slots=True)
class CanonicalFunctionToolDefinition:
    """Canonical tool definition: wire schema plus typed dispatch requirements."""

    wire_schema: dict[str, Any]
    dispatch_requirements: ToolDispatchRequirements

    @property
    def requires_strict_argument_conformance(self) -> bool:
        return self.dispatch_requirements.requires_strict_argument_conformance


_STRICT_FUNCTION_TOOL_NAMES: frozenset[str] = frozenset()


class StrictToolArgumentConformanceError(RuntimeError):
    """Tool schema requires provider-enforced argument conformance that the adapter lacks."""


def register_strict_function_tool_name(name: str) -> None:
    """Register a platform function tool name that requires strict argument conformance."""
    global _STRICT_FUNCTION_TOOL_NAMES
    _STRICT_FUNCTION_TOOL_NAMES = _STRICT_FUNCTION_TOOL_NAMES | frozenset({name})


def _function_tool_name(tool: Mapping[str, Any]) -> str | None:
    if tool.get("type") != "function":
        return None
    fn = tool.get("function")
    if not isinstance(fn, Mapping):
        return None
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    return name


def function_tool_dispatch_requirements(
    tool: Mapping[str, Any],
) -> ToolDispatchRequirements:
    """Resolve typed dispatch requirements for a canonical function tool wire schema."""
    name = _function_tool_name(tool)
    if name is not None and name in _STRICT_FUNCTION_TOOL_NAMES:
        return ToolDispatchRequirements(
            argument_conformance=ToolArgumentConformance.STRICT,
        )
    return ToolDispatchRequirements()


def function_tool_requires_strict_argument_conformance(
    tool: Mapping[str, Any],
) -> bool:
    """Return whether a canonical function tool requires provider-side strict conformance."""
    return function_tool_dispatch_requirements(tool).requires_strict_argument_conformance


def tools_schema_requires_strict_argument_conformance(
    tools_schema: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether any tool in the schema list requires strict argument conformance."""
    return any(
        function_tool_requires_strict_argument_conformance(tool) for tool in tools_schema
    )


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
