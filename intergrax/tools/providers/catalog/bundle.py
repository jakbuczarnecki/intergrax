# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.catalog.contracts import (
    CatalogDescribeToolInput,
    CatalogDescribeToolOutput,
    CatalogListToolsInput,
    CatalogListToolsOutput,
)
from intergrax.tools.providers.catalog.handlers import CatalogDescribeToolHandler, CatalogListToolsHandler
from intergrax.tools.providers.catalog.service import CATALOG_DESCRIBE_TOOL_TOOL_ID, CATALOG_LIST_TOOLS_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CATALOG_BUNDLE_ID = "catalog"
CATALOG_TOOL_IDS: tuple[str, ...] = (CATALOG_LIST_TOOLS_TOOL_ID, CATALOG_DESCRIBE_TOOL_TOOL_ID)


def register_catalog_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CATALOG_LIST_TOOLS_TOOL_ID,
            name=CATALOG_LIST_TOOLS_TOOL_ID,
            description="List tool_ids available in the current ToolRegistry (optional category/tag filters).",
            description_short="List catalog tools.",
            input_schema=CatalogListToolsInput,
            output_schema=CatalogListToolsOutput,
            error_mapping={},
            side_effects=False,
            category="catalog",
            risk_level=ToolRiskLevel.LOW,
            tags=("catalog", "introspection", "dx"),
        ),
        CatalogListToolsHandler(ctx, registry),
    )
    registry.register(
        ToolContract(
            tool_id=CATALOG_DESCRIBE_TOOL_TOOL_ID,
            name=CATALOG_DESCRIBE_TOOL_TOOL_ID,
            description="Describe one catalog tool contract including JSON schemas for LLM planning.",
            description_short="Describe catalog tool.",
            input_schema=CatalogDescribeToolInput,
            output_schema=CatalogDescribeToolOutput,
            error_mapping={},
            side_effects=False,
            category="catalog",
            risk_level=ToolRiskLevel.LOW,
            tags=("catalog", "introspection", "dx"),
        ),
        CatalogDescribeToolHandler(ctx, registry),
    )
