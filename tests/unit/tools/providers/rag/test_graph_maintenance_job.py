# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

import pytest

from intergrax.integrations.contracts.workflow_orchestrator import WorkflowRunHandle, WorkflowRunStatus
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.tools.providers.rag.graph_maintenance_contracts import RagScheduleGraphMaintenanceJobInput
from intergrax.tools.providers.rag.graph_maintenance_service import perform_rag_schedule_graph_maintenance_job
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.triggered: list[dict[str, str]] = []
        self._runs: list[WorkflowRunHandle] = []

    def trigger_run(self, workflow_id: str, *, parameters: dict[str, str] | None = None) -> WorkflowRunHandle:
        params = dict(parameters or {})
        handle = WorkflowRunHandle(
            run_id=f"run-{len(self._runs) + 1}",
            status="pending",
            url=f"https://wf/{workflow_id}",
            metadata={"idempotency_key": params.get("idempotency_key", "")},
        )
        self.triggered.append(params)
        self._runs.append(handle)
        return handle

    def poll_status(self, run_id: str) -> WorkflowRunStatus:
        return WorkflowRunStatus(run_id=run_id, status="pending")

    def fetch_logs(self, run_id: str, *, tail_lines: int = 200) -> str:
        return f"log:{run_id}"

    def list_runs(self, *, workflow_id: str = "", limit: int = 20) -> Sequence[WorkflowRunHandle]:
        del workflow_id, limit
        return list(self._runs)

    def cancel_run(self, run_id: str) -> WorkflowRunStatus:
        return WorkflowRunStatus(run_id=run_id, status="cancelled")


def test_schedule_graph_maintenance_job_triggers_workflow() -> None:
    orchestrator = _FakeOrchestrator()
    profile = RagProfile(graph_rag_enabled=True, graph_maintenance_workflow_id="rag-graph-maint")
    ctx = ToolWiringContext(workflow_orchestrator=orchestrator, rag_profile=profile)
    out = perform_rag_schedule_graph_maintenance_job(
        ctx,
        RagScheduleGraphMaintenanceJobInput(mode="orphan_prune", tenant_id="tenant-1"),
    )
    assert out.used is True
    assert out.reason == "scheduled"
    assert orchestrator.triggered[0]["job_type"] == "rag.graph_maintenance"
    assert orchestrator.triggered[0]["mode"] == "orphan_prune"


def test_schedule_graph_maintenance_job_is_idempotent() -> None:
    orchestrator = _FakeOrchestrator()
    profile = RagProfile(graph_rag_enabled=True)
    ctx = ToolWiringContext(workflow_orchestrator=orchestrator, rag_profile=profile)
    params = RagScheduleGraphMaintenanceJobInput(mode="stale_edge_prune", tenant_id="tenant-2")
    first = perform_rag_schedule_graph_maintenance_job(ctx, params)
    second = perform_rag_schedule_graph_maintenance_job(ctx, params)
    assert first.used and second.used
    assert second.reason == "idempotent_reuse"
    assert len(orchestrator.triggered) == 1
