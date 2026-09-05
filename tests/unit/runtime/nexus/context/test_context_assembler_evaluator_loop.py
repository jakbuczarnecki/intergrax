# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.nexus.context.context_assembler import collect_dependency_records
from intergrax.runtime.nexus.execution.evaluator_loop_metadata import (
    set_evaluator_loop_iteration,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode


def test_evaluator_loop_revision_includes_self_prior_output() -> None:
    node = ExecutionNode(
        node_id="node_incident_investigator",
        agent_id="incident_investigator",
        capability="incident_investigation.investigate",
    )
    prior = AgentExecutionResult(
        agent_id="incident_investigator",
        run_id="run_test",
        status=AgentExecutionStatus.COMPLETED,
        summary="initial pass",
        structured_data={
            "domain_summary": {
                "evidence_nodes": [
                    {"evidence_id": "evidence.workload.line4.incident_window", "payload": {}},
                ],
            },
        },
    )
    prior_outputs = {node.node_id: prior}

    without_revision, _, _ = collect_dependency_records(
        node,
        prior_outputs,
        policy=TaskContextAssemblyOptions(),
        shared_version=1,
    )
    assert without_revision == []

    set_evaluator_loop_iteration(node, 1)
    with_revision, _, _ = collect_dependency_records(
        node,
        prior_outputs,
        policy=TaskContextAssemblyOptions(),
        shared_version=1,
    )
    assert len(with_revision) == 1
    assert with_revision[0].node_id == node.node_id
    assert with_revision[0].structured_data["domain_summary"]["evidence_nodes"]
