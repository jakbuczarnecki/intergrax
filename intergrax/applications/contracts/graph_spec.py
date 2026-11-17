# © Artur Czarnecki. All rights reserved.

"""Declarative multi-agent topology for Tier-3 applications (Phase H-APP.3.2)."""

from __future__ import annotations

from intergrax.contracts.application_graph_topology import (
    ApplicationGraphTopology,
    CodeCraftGraphBinding,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
)
from intergrax.contracts.evaluator_loop import EvaluatorLoopGraphBinding


class ApplicationGraphSpec(ApplicationGraphTopology):
    """
    Validated application graph — nodes must exist on the manifest roster.

    Used by host orchestration backend spec resolution for Execution Engine materialization.
    The topology fields and the unique-node validator live on the neutral Tier-0 base model.
    """

    def validate_against_roster(self, bindings: list[AgentBinding]) -> None:  # noqa: F821
        """Raise ``ValueError`` when graph references unknown agents."""
        from intergrax.applications.contracts.manifest import AgentBinding  # noqa: F401 — roster typing

        enabled = [b for b in bindings if b.enabled]
        known: set[str] = set()
        for binding in enabled:
            contract_id = binding.contract_id
            if contract_id:
                known.add(contract_id.strip())
            if binding.agent_type is not None:
                known.add(binding.agent_type.__name__)
            elif binding.import_path:
                known.add(binding.import_path.rsplit(".", 1)[-1])

        for node in self.nodes:
            if node.agent_id not in known and (node.contract_id is None or node.contract_id not in known):
                raise ValueError(f"ApplicationGraphSpec node {node.agent_id!r} not found on manifest roster")

        roster_ids = self.roster_agent_ids()
        for edge in self.edges:
            if edge.source_agent_id not in roster_ids:
                raise ValueError(f"Graph edge source {edge.source_agent_id!r} missing from nodes")
            if edge.target_agent_id not in roster_ids:
                raise ValueError(f"Graph edge target {edge.target_agent_id!r} missing from nodes")


__all__ = [
    "ApplicationGraphSpec",
    "CodeCraftGraphBinding",
    "EvaluatorLoopGraphBinding",
    "GraphEdge",
    "GraphEdgeKind",
    "GraphNode",
]
