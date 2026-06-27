# © Artur Czarnecki. All rights reserved.

"""Graph editor visualization helpers (AUDIT-IDEAL-27.4)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec


class GraphEditorArtifact(BaseModel):
    schema_version: str = "1.0.0"
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    mermaid: str
    valid: bool


def render_graph_mermaid(spec: ApplicationGraphSpec) -> str:
    """Render a Mermaid diagram for a validated application graph spec."""
    lines = ["graph TD"]
    for node in spec.nodes:
        lines.append(f'  {node.agent_id}["{node.agent_id}"]')
    for edge in spec.edges:
        arrow = "-->" if edge.kind.value == "depends_on" else "-.->"
        lines.append(f"  {edge.source_agent_id} {arrow} {edge.target_agent_id}")
    return "\n".join(lines)


def build_graph_editor_artifact(spec: ApplicationGraphSpec) -> GraphEditorArtifact:
    """Validate and visualize a Tier-3 application graph."""
    mermaid = render_graph_mermaid(spec)
    valid = bool(spec.nodes) and all(
        any(edge.source_agent_id == node.agent_id or edge.target_agent_id == node.agent_id for edge in spec.edges)
        or len(spec.nodes) == 1
        for node in spec.nodes
    )
    return GraphEditorArtifact(
        node_count=len(spec.nodes),
        edge_count=len(spec.edges),
        mermaid=mermaid,
        valid=valid,
    )
