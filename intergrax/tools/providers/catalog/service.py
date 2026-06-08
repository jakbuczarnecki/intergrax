# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.providers.catalog.contracts import (
    CatalogDescribeToolInput,
    CatalogDescribeToolOutput,
    CatalogListToolsInput,
    CatalogListToolsOutput,
    CatalogToolSummary,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CATALOG_LIST_TOOLS_TOOL_ID = "catalog.list_tools"
CATALOG_DESCRIBE_TOOL_TOOL_ID = "catalog.describe_tool"


def _require_registry(ctx: ToolWiringContext, registry: ToolRegistry | None) -> ToolRegistry:
    if registry is not None:
        return registry
    bound = ctx.extras.get("tool_registry")
    if isinstance(bound, ToolRegistry):
        return bound
    raise RuntimeError("tool_registry_not_configured")


def _summarize(contract: ToolContract) -> CatalogToolSummary:
    return CatalogToolSummary(
        tool_id=contract.tool_id,
        name=contract.name,
        description_short=contract.description_short or "",
        category=contract.category or "",
        risk_level=contract.risk_level.value if contract.risk_level else "",
        side_effects=bool(contract.side_effects),
    )


def catalog_list_tools(
    ctx: ToolWiringContext,
    params: CatalogListToolsInput,
    *,
    registry: ToolRegistry | None = None,
) -> CatalogListToolsOutput:
    target = _require_registry(ctx, registry)
    category = params.category.strip().lower()
    tag = params.tag.strip().lower()
    summaries: list[CatalogToolSummary] = []
    for tool_id in sorted(target.tool_ids()):
        contract = target.get(tool_id).contract
        if category and (contract.category or "").lower() != category:
            continue
        contract_tags = {item.lower() for item in (contract.tags or ())}
        if tag and tag not in contract_tags:
            continue
        summaries.append(_summarize(contract))
    return CatalogListToolsOutput(tools=summaries, total=len(summaries))


def catalog_describe_tool(
    ctx: ToolWiringContext,
    params: CatalogDescribeToolInput,
    *,
    registry: ToolRegistry | None = None,
) -> CatalogDescribeToolOutput:
    target = _require_registry(ctx, registry)
    tool_id = params.tool_id.strip()
    if not target.has(tool_id):
        return CatalogDescribeToolOutput(found=False, tool_id=tool_id)
    contract = target.get(tool_id).contract
    input_schema = contract.input_schema.model_json_schema() if contract.input_schema else {}
    output_schema = contract.output_schema.model_json_schema() if contract.output_schema else {}
    return CatalogDescribeToolOutput(
        found=True,
        tool_id=contract.tool_id,
        name=contract.name,
        description=contract.description,
        description_short=contract.description_short or "",
        category=contract.category or "",
        risk_level=contract.risk_level.value if contract.risk_level else "",
        side_effects=bool(contract.side_effects),
        input_schema=input_schema,
        output_schema=output_schema,
        tags=list(contract.tags or ()),
    )
