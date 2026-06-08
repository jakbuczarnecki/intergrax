# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed graph node diagnostics for observability spine (OBS-BUS-3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

GRAPH_NODE_STEP_START = "graph.node_start"
GRAPH_NODE_STEP_COMPLETE = "graph.node_complete"


@dataclass(frozen=True)
class GraphNodeDiagV1(DiagnosticPayload):
    node_id: str
    status: str
    agent_id: str = ""
    capability: str = ""

    def redact(self) -> GraphNodeDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.graph.node"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "agent_id": self.agent_id,
            "capability": self.capability,
        }
