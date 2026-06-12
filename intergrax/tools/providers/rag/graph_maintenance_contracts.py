# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

GraphMaintenanceMode = Literal["orphan_prune", "stale_edge_prune", "full_reindex"]


class RagScheduleGraphMaintenanceJobInput(BaseModel):
    mode: GraphMaintenanceMode = Field(
        default="orphan_prune",
        description="Maintenance operation: orphan_prune | stale_edge_prune | full_reindex.",
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Orchestrator workflow id; defaults to RagProfile.graph_maintenance_workflow_id.",
    )
    idempotency_key: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None


class RagScheduleGraphMaintenanceJobOutput(BaseModel):
    used: bool
    run_id: str = ""
    status: str = ""
    url: str = ""
    idempotency_key: str = ""
    reason: str = ""
