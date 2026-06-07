# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.graph.contracts import (
    GraphGetNodeInput,
    GraphNodeOutput,
    GraphRunQueryInput,
    GraphRunQueryOutput,
)
from intergrax.tools.providers.graph.service import graph_get_node, graph_run_query


class GraphRunQueryHandler(ServiceToolHandler[GraphRunQueryInput, GraphRunQueryOutput]):
    _service = graph_run_query


class GraphGetNodeHandler(ServiceToolHandler[GraphGetNodeInput, GraphNodeOutput]):
    _service = graph_get_node
