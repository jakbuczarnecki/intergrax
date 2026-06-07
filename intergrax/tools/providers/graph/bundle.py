# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.graph.contracts import (
    GraphGetNodeInput,
    GraphNodeOutput,
    GraphRunQueryInput,
    GraphRunQueryOutput,
)
from intergrax.tools.providers.graph.handlers import GraphGetNodeHandler, GraphRunQueryHandler
from intergrax.tools.providers.graph.service import GRAPH_GET_NODE_TOOL_ID, GRAPH_RUN_QUERY_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

GRAPH_BUNDLE_ID = "graph"
GRAPH_TOOL_IDS: tuple[str, ...] = (GRAPH_RUN_QUERY_TOOL_ID, GRAPH_GET_NODE_TOOL_ID)


def register_graph_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=GRAPH_RUN_QUERY_TOOL_ID,
            name=GRAPH_RUN_QUERY_TOOL_ID,
            description="Execute a property-graph query (Cypher or vendor-neutral) on the configured graph store.",
            description_short="Run graph query.",
            input_schema=GraphRunQueryInput,
            output_schema=GraphRunQueryOutput,
            error_mapping={},
            side_effects=False,
            category="graph",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("graph", "query"),
        ),
        GraphRunQueryHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=GRAPH_GET_NODE_TOOL_ID,
            name=GRAPH_GET_NODE_TOOL_ID,
            description="Fetch a graph node by internal id.",
            description_short="Get graph node.",
            input_schema=GraphGetNodeInput,
            output_schema=GraphNodeOutput,
            error_mapping={},
            side_effects=False,
            category="graph",
            risk_level=ToolRiskLevel.LOW,
            tags=("graph", "node"),
        ),
        GraphGetNodeHandler(ctx),
    )
