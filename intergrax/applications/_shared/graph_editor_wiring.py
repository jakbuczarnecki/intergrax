# © Artur Czarnecki. All rights reserved.

"""Visual graph editor wiring for product hosts (AUDIT-IDEAL-27.4)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphEdge, GraphNode
from intergrax.runtime.architecture.graph_visualization import build_graph_editor_artifact


class GraphEditorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ApplicationGraphSpec


def create_graph_editor_router(*, enabled: bool = True) -> APIRouter:
    router = APIRouter(prefix="/ops/graph", tags=["graph-editor"])

    @router.post("/render")
    def render_graph(request: GraphEditorRequest) -> dict[str, object]:
        if not enabled:
            return {"enabled": False}
        artifact = build_graph_editor_artifact(request.spec)
        return {
            "enabled": True,
            "valid": artifact.valid,
            "node_count": artifact.node_count,
            "edge_count": artifact.edge_count,
            "mermaid": artifact.mermaid,
        }

    @router.get("/template")
    def graph_template() -> dict[str, object]:
        if not enabled:
            return {"enabled": False}
        spec = ApplicationGraphSpec(
            nodes=[
                GraphNode(agent_id="producer", contract_id="producer"),
                GraphNode(agent_id="evaluator", contract_id="evaluator"),
            ],
            edges=[
                GraphEdge(source_agent_id="producer", target_agent_id="evaluator"),
            ],
        )
        artifact = build_graph_editor_artifact(spec)
        return {"enabled": True, "spec": spec.model_dump(), "mermaid": artifact.mermaid}

    return router


@dataclass(frozen=True, slots=True)
class GraphEditorWiring:
    enabled: bool
    router: APIRouter | None


def resolve_graph_editor_wiring(env: ApplicationEnvironmentProfile) -> GraphEditorWiring:
    """Mount graph editor HTTP routes on product hosts when enabled."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return GraphEditorWiring(enabled=False, router=None)
    if not env.features.graph_editor_enabled:
        return GraphEditorWiring(enabled=False, router=None)
    return GraphEditorWiring(enabled=True, router=create_graph_editor_router(enabled=True))
