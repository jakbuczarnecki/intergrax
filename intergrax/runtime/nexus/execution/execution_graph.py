# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.delegation import DelegationSpec


class ExecutionNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionNode(BaseModel):
    """Node in a Nexus execution graph (§24)."""

    node_id: str
    agent_id: Optional[str] = None
    capability: Optional[str] = None
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)
    status: ExecutionNodeStatus = ExecutionNodeStatus.PENDING
    batch_index: int = 0
    execution_result: Optional[AgentExecutionResult] = None
    delegation: Optional[DelegationSpec] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionGraph(BaseModel):
    """Task execution graph with dependency edges (§24)."""

    graph_id: str
    task_id: str
    nodes: List[ExecutionNode] = Field(default_factory=list)

    def node_by_id(self, node_id: str) -> ExecutionNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"ExecutionNode not found: {node_id}")

    def batches(self) -> List[List[ExecutionNode]]:
        """Topological batches: nodes in a batch may run in parallel (§25)."""
        if not self.nodes:
            return []

        node_map = {n.node_id: n for n in self.nodes}
        in_degree: Dict[str, int] = {n.node_id: len(n.depends_on) for n in self.nodes}
        batches: List[List[ExecutionNode]] = []
        batch_index = 0

        while in_degree:
            ready_ids = [nid for nid, deg in in_degree.items() if deg == 0]
            if not ready_ids:
                ready_ids = list(in_degree.keys())

            batch: List[ExecutionNode] = []
            for nid in ready_ids:
                node = node_map[nid]
                node.batch_index = batch_index
                batch.append(node)
                del in_degree[nid]

            batches.append(batch)
            batch_index += 1

            for completed_id in ready_ids:
                for nid in list(in_degree.keys()):
                    if completed_id in node_map[nid].depends_on:
                        in_degree[nid] -= 1

        return batches
