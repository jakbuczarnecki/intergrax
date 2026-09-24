# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Construct CodeCraft qualified capability execution handler (UCA-6C-R4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionPort,
)
from intergrax.runtime.codecraft.qualified_capability_execution_handler import (
    CodeCraftQualifiedCapabilityExecutionHandler,
)
from intergrax.runtime.codecraft.wiring_bound_capability_execution import (
    WiringCodeCraftBoundCapabilityExecution,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class CodeCraftQualifiedCapabilityExecutionComposition:
    """Handler + wiring artifacts for downstream qualified capability binding."""

    handler: CodeCraftQualifiedCapabilityExecutionHandler
    tool_wiring_context: ToolWiringContext
    execution_port: CodeCraftBoundCapabilityExecutionPort


def build_codecraft_qualified_capability_execution_composition(
    wiring_context: ToolWiringContext,
    *,
    execution_port: CodeCraftBoundCapabilityExecutionPort | None = None,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker | None = None,
    side_effect_recorder: list[str] | None = None,
) -> CodeCraftQualifiedCapabilityExecutionComposition:
    port = execution_port or WiringCodeCraftBoundCapabilityExecution(
        wiring_context,
        catalog_tool_invoker=catalog_tool_invoker,
    )
    handler = CodeCraftQualifiedCapabilityExecutionHandler(
        execution_port=port,
        side_effect_recorder=side_effect_recorder,
    )
    return CodeCraftQualifiedCapabilityExecutionComposition(
        handler=handler,
        tool_wiring_context=wiring_context,
        execution_port=port,
    )


def build_codecraft_qualified_capability_execution_handler(
    wiring_context: ToolWiringContext,
    *,
    execution_port: CodeCraftBoundCapabilityExecutionPort | None = None,
    catalog_tool_invoker: ExecutionBoundCatalogToolInvoker | None = None,
    side_effect_recorder: list[str] | None = None,
) -> CodeCraftQualifiedCapabilityExecutionHandler:
    return build_codecraft_qualified_capability_execution_composition(
        wiring_context,
        execution_port=execution_port,
        catalog_tool_invoker=catalog_tool_invoker,
        side_effect_recorder=side_effect_recorder,
    ).handler


__all__ = [
    "CodeCraftQualifiedCapabilityExecutionComposition",
    "build_codecraft_qualified_capability_execution_composition",
    "build_codecraft_qualified_capability_execution_handler",
]
