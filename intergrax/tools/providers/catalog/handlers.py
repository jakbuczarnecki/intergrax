# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.catalog.contracts import (
    CatalogDescribeToolInput,
    CatalogDescribeToolOutput,
    CatalogListToolsInput,
    CatalogListToolsOutput,
)
from intergrax.tools.providers.catalog.service import catalog_describe_tool, catalog_list_tools
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.tools.tool_executor import ToolHandler


class CatalogListToolsHandler(ToolHandler[CatalogListToolsInput, CatalogListToolsOutput]):
    def __init__(self, ctx: ToolWiringContext, registry: ToolRegistry) -> None:
        self._ctx = ctx
        self._registry = registry

    def execute(self, request: ToolExecutionRequest[CatalogListToolsInput]) -> CatalogListToolsOutput:
        return catalog_list_tools(self._ctx, request.input, registry=self._registry)


class CatalogDescribeToolHandler(ToolHandler[CatalogDescribeToolInput, CatalogDescribeToolOutput]):
    def __init__(self, ctx: ToolWiringContext, registry: ToolRegistry) -> None:
        self._ctx = ctx
        self._registry = registry

    def execute(self, request: ToolExecutionRequest[CatalogDescribeToolInput]) -> CatalogDescribeToolOutput:
        return catalog_describe_tool(self._ctx, request.input, registry=self._registry)
