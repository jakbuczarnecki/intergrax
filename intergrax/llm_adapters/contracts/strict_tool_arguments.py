# © Artur Czarnecki. All rights reserved.

"""Provider-neutral strict tool argument conformance contract (DS-E2E-13B)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class CanonicalFunctionToolFunctionSchema(TypedDict):
    """Provider-neutral OpenAI-style function tool payload."""

    name: str
    description: str
    parameters: dict[str, object]


class CanonicalFunctionToolWireSchema(TypedDict):
    """Provider-neutral function-tool wire schema shared across adapters."""

    type: str
    function: CanonicalFunctionToolFunctionSchema


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

    wire_schema: CanonicalFunctionToolWireSchema
    dispatch_requirements: ToolDispatchRequirements

    @property
    def requires_strict_argument_conformance(self) -> bool:
        return self.dispatch_requirements.requires_strict_argument_conformance


class StrictToolArgumentConformanceError(RuntimeError):
    """Tool schema requires provider-enforced argument conformance that the adapter lacks."""


def aligned_tool_dispatch_requirements(
    tools_schema: Sequence[Mapping[str, Any]],
    *,
    tool_dispatch_requirements: Sequence[ToolDispatchRequirements] | None = None,
) -> tuple[ToolDispatchRequirements, ...]:
    """Return per-tool dispatch requirements aligned with ``tools_schema`` indices."""
    if tool_dispatch_requirements is None:
        return tuple(ToolDispatchRequirements() for _ in tools_schema)
    if len(tool_dispatch_requirements) != len(tools_schema):
        raise ValueError(
            "tool_dispatch_requirements length must match tools_schema length"
        )
    return tuple(tool_dispatch_requirements)


def tools_schema_requires_strict_argument_conformance(
    tools_schema: Sequence[Mapping[str, Any]],
    *,
    tool_dispatch_requirements: Sequence[ToolDispatchRequirements] | None = None,
) -> bool:
    """Return whether any tool in the schema list requires strict argument conformance."""
    aligned = aligned_tool_dispatch_requirements(
        tools_schema,
        tool_dispatch_requirements=tool_dispatch_requirements,
    )
    return any(
        requirements.requires_strict_argument_conformance for requirements in aligned
    )


def assert_strict_tool_argument_conformance_supported(
    adapter: LLMAdapter,
    tools_schema: Sequence[Mapping[str, Any]],
    *,
    tool_dispatch_requirements: Sequence[ToolDispatchRequirements] | None = None,
) -> None:
    """Fail closed when strict tools are dispatched to an adapter without capability."""
    if not tools_schema_requires_strict_argument_conformance(
        tools_schema,
        tool_dispatch_requirements=tool_dispatch_requirements,
    ):
        return
    if adapter.supports_strict_tool_argument_conformance():
        return
    raise StrictToolArgumentConformanceError(
        "tools schema requires provider-enforced strict argument conformance but "
        f"{adapter.__class__.__name__} does not support it"
    )
