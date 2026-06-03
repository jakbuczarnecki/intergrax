# © Artur Czarnecki. All rights reserved.

"""Declarative multi-agent topology for Tier-3 applications (Phase H-APP.3.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

class GraphEdgeKind(str, Enum):
    DEPENDS_ON = "depends_on"
    DELEGATES_TO = "delegates_to"


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_agent_id: str
    target_agent_id: str
    kind: GraphEdgeKind = GraphEdgeKind.DEPENDS_ON


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    contract_id: str | None = None


class ApplicationGraphSpec(BaseModel):
    """
    Validated application graph — nodes must exist on the manifest roster.

    Used by :func:`~intergrax.applications._shared.nexus_factory.build_nexus_loop_from_environment`.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    def roster_agent_ids(self) -> frozenset[str]:
        return frozenset(node.agent_id for node in self.nodes)

    def validate_against_roster(self, bindings: list["AgentBinding"]) -> None:
        """Raise ``ValueError`` when graph references unknown agents."""
        from intergrax.applications.contracts.manifest import AgentBinding  # noqa: F401 — roster typing
        enabled = [b for b in bindings if b.enabled]
        known: set[str] = set()
        for binding in enabled:
            contract_id = binding.contract_id
            if contract_id:
                known.add(contract_id.strip())
            known.add(binding.resolved_agent_type().__name__)
            if binding.import_path:
                known.add(binding.import_path.rsplit(".", 1)[-1])

        for node in self.nodes:
            if node.agent_id not in known and (node.contract_id is None or node.contract_id not in known):
                raise ValueError(
                    f"ApplicationGraphSpec node {node.agent_id!r} not found on manifest roster"
                )

        roster_ids = self.roster_agent_ids()
        for edge in self.edges:
            if edge.source_agent_id not in roster_ids:
                raise ValueError(f"Graph edge source {edge.source_agent_id!r} missing from nodes")
            if edge.target_agent_id not in roster_ids:
                raise ValueError(f"Graph edge target {edge.target_agent_id!r} missing from nodes")

    @model_validator(mode="after")
    def _unique_nodes(self) -> ApplicationGraphSpec:
        seen: set[str] = set()
        for node in self.nodes:
            if node.agent_id in seen:
                raise ValueError(f"duplicate graph node agent_id: {node.agent_id!r}")
            seen.add(node.agent_id)
        return self
