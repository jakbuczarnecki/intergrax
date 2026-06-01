# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.delegation import DelegationSpec
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode


@pytest.mark.unit
def test_execution_node_carries_delegation_spec() -> None:
    spec = DelegationSpec(child_agent_id="research", isolated_memory_namespace="")
    node = ExecutionNode(
        node_id="n1",
        agent_id="research",
        delegation=spec,
    )
    assert node.delegation is not None
    namespace = node.delegation.resolved_memory_namespace(task_id="t1", node_id="n1")
    assert namespace == "t1/delegation/n1"
