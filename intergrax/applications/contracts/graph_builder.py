# © Artur Czarnecki. All rights reserved.

"""Fluent builder for Tier-3 multi-agent topology (Phase DX-2.2)."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
)


class AgentGraph:
    """LangGraph-style fluent API mapping to :class:`ApplicationGraphSpec`."""

    def __init__(self) -> None:
        self._nodes: list[GraphNode] = []
        self._edges: list[GraphEdge] = []
        self._default_agent_id: str | None = None
        self._retry_on_error: int | None = None

    def add(self, agent_type: type[Agent], *, agent_id: str | None = None) -> AgentGraph:
        resolved_id = agent_id or agent_type.__name__
        raw_contract = agent_type.__dict__.get("contract_id")
        contract_id = raw_contract if isinstance(raw_contract, str) else None
        self._nodes.append(GraphNode(agent_id=resolved_id, contract_id=contract_id))
        return self

    def default(self, agent_type: type[Agent], *, agent_id: str | None = None) -> AgentGraph:
        resolved_id = agent_id or agent_type.__name__
        self._default_agent_id = resolved_id
        if not any(node.agent_id == resolved_id for node in self._nodes):
            self.add(agent_type, agent_id=resolved_id)
        return self

    def edge(
        self,
        source: str,
        target: str,
        *,
        kind: GraphEdgeKind = GraphEdgeKind.DEPENDS_ON,
    ) -> AgentGraph:
        self._edges.append(
            GraphEdge(
                source_agent_id=source,
                target_agent_id=target,
                kind=kind,
            )
        )
        return self

    def delegates_to(self, source: str, target: str) -> AgentGraph:
        return self.edge(source, target, kind=GraphEdgeKind.DELEGATES_TO)

    def on_error(self, *, retry: int) -> AgentGraph:
        if retry < 0:
            raise ValueError("retry must be non-negative")
        self._retry_on_error = retry
        return self

    def build(self) -> ApplicationGraphSpec:
        return ApplicationGraphSpec(
            nodes=list(self._nodes),
            edges=list(self._edges),
            retry_on_error=self._retry_on_error,
        )

    @property
    def default_agent_id(self) -> str | None:
        return self._default_agent_id

    @property
    def retry_on_error(self) -> int | None:
        return self._retry_on_error
