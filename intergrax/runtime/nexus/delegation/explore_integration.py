# © Artur Czarnecki. All rights reserved.

"""Explore delegation context injection for graph executor (MEM-DEPTH-4.2)."""

from __future__ import annotations

from intergrax.contracts.delegation import DelegationSpec
from intergrax.runtime.nexus.delegation.explore_runner import ExploreDelegationRunner
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


def apply_explore_delegation_context(
    request: RuntimeRequest,
    delegation: DelegationSpec,
    *,
    task_id: str,
    node_id: str,
) -> None:
    """Run explore synthesis and attach bounded payload to child request metadata."""
    if delegation.explore is None:
        return
    runner = ExploreDelegationRunner(delegation.explore)
    result = runner.run(
        delegation,
        task_id=task_id,
        node_id=node_id,
    )
    request.metadata["explore_synthesis"] = result.synthesis_text
    request.metadata["explore_memory_namespace"] = result.memory_namespace
    request.metadata["explore_findings_count"] = len(result.findings)
    request.metadata["explore_delegation"] = True
