# © Artur Czarnecki. All rights reserved.

"""Execution graph cycle detection (Phase FLOW-6)."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionGraphCycleError,
    ExecutionNode,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_batches_raises_on_cycle() -> None:
    graph = ExecutionGraph(
        graph_id="g1",
        task_id="task_cycle",
        nodes=[
            ExecutionNode(node_id="a", agent_id="A", depends_on=["b"]),
            ExecutionNode(node_id="b", agent_id="B", depends_on=["a"]),
        ],
    )
    with pytest.raises(ExecutionGraphCycleError):
        graph.batches()
