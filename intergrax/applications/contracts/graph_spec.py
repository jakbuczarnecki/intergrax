# © Artur Czarnecki. All rights reserved.

"""Declarative multi-agent topology for Tier-3 applications (Phase H-APP.3.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec

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


class EvaluatorLoopGraphBinding(BaseModel):
    """Standard evaluator-loop topology for product graph specs (AUDIT-IDEAL-10.1)."""

    model_config = ConfigDict(extra="forbid")

    producer_agent_id: str
    evaluator_agent_id: str
    revise_agent_id: str
    spec: EvaluatorLoopSpec


class CodeCraftGraphBinding(BaseModel):
    """Optional code craft node in application graph specs (ECC-5)."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    goal_template: str = "Synthesize helper for task"
    max_iterations: int = Field(default=8, ge=1, le=64)
    promote_on_success: bool = True


class ApplicationGraphSpec(BaseModel):
    """
    Validated application graph — nodes must exist on the manifest roster.

    Used by :func:`~intergrax.applications._shared.nexus_factory.build_nexus_loop_from_environment`.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    retry_on_error: int | None = Field(default=None, ge=0, le=32)
    trigger_capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "When non-empty, graph seeding applies only for these task capabilities. "
            "When empty, seeding uses pipeline_capability_suffix convention (ORCH-CONFIG.2)."
        ),
    )
    pipeline_capability_suffix: str = Field(
        default=".pipeline",
        min_length=1,
        description="Capability suffix that triggers graph seed when trigger_capabilities is empty.",
    )
    evaluator_loop: EvaluatorLoopGraphBinding | None = None
    codecraft_node: CodeCraftGraphBinding | None = None

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
            if binding.agent_type is not None or binding.import_path is not None:
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
