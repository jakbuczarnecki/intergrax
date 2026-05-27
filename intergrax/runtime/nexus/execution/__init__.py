# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)

if TYPE_CHECKING:
    from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor

__all__ = [
    "ExecutionGraph",
    "ExecutionNode",
    "ExecutionNodeStatus",
    "GraphExecutor",
    "plan_to_execution_graph",
]


def __getattr__(name: str):
    if name == "GraphExecutor":
        from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor

        return GraphExecutor
    if name == "plan_to_execution_graph":
        from intergrax.runtime.nexus.execution.graph_builder import plan_to_execution_graph

        return plan_to_execution_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
