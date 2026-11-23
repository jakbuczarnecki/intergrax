# © Artur Czarnecki. All rights reserved.

"""Neutral application graph topology (Tier-0 contract; Tier-3 ``ApplicationGraphSpec`` subclasses it)."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.evaluator_loop import EvaluatorLoopGraphBinding


class GraphEdgeKind(str, Enum):  # noqa: UP042
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


class CodeCraftGraphBinding(BaseModel):
    """Optional code craft node in application graph specs (ECC-5)."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    goal_template: str = "Synthesize helper for task"
    max_iterations: int = Field(default=8, ge=1, le=64)
    promote_on_success: bool = True


class ApplicationGraphTopology(BaseModel):
    """Declarative multi-agent topology consumed by Tier-1 planning and classification."""

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

    @model_validator(mode="after")
    def _unique_nodes(self) -> ApplicationGraphTopology:
        seen: set[str] = set()
        for node in self.nodes:
            if node.agent_id in seen:
                raise ValueError(f"duplicate graph node agent_id: {node.agent_id!r}")
            seen.add(node.agent_id)
        return self


@runtime_checkable
class HostGraphTopologySource(Protocol):
    """Live read access to the current graph topology (planner reads it at plan time)."""

    def current_graph_topology(self) -> ApplicationGraphTopology | None: ...


__all__ = [
    "ApplicationGraphTopology",
    "CodeCraftGraphBinding",
    "GraphEdge",
    "GraphEdgeKind",
    "GraphNode",
    "HostGraphTopologySource",
]
