# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.graph_store import GraphStore
from intergrax.tools.providers.graph.contracts import (
    GraphGetNodeInput,
    GraphNodeOutput,
    GraphRunQueryInput,
    GraphRunQueryOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

GRAPH_RUN_QUERY_TOOL_ID = "graph.run_query"
GRAPH_GET_NODE_TOOL_ID = "graph.get_node"


def _require_graph(ctx: ToolWiringContext) -> GraphStore:
    store = ctx.graph_store
    if store is None:
        raise RuntimeError("graph_store_not_configured")
    return store


def graph_run_query(ctx: ToolWiringContext, params: GraphRunQueryInput) -> GraphRunQueryOutput:
    result = _require_graph(ctx).run_query(params.statement.strip(), parameters=dict(params.parameters))
    records = [dict(item) for item in result.records]
    return GraphRunQueryOutput(
        records=records,
        summary=result.summary,
        record_count=len(records),
    )


def graph_get_node(ctx: ToolWiringContext, params: GraphGetNodeInput) -> GraphNodeOutput:
    node = _require_graph(ctx).get_node(params.node_id.strip())
    if node is None:
        return GraphNodeOutput(id=params.node_id.strip(), found=False)
    return GraphNodeOutput(
        id=node.id,
        labels=list(node.labels),
        properties=dict(node.properties),
        found=True,
    )
