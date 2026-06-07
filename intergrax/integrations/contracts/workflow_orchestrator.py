# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Workflow orchestrator integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class WorkflowRunHandle(BaseModel):
    """Submitted workflow run reference."""

    run_id: str
    status: str = "pending"
    url: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class WorkflowRunStatus(BaseModel):
    """Polled workflow run status."""

    run_id: str
    status: str = ""
    conclusion: str = ""
    logs_uri: str = ""


@runtime_checkable
class WorkflowOrchestratorBackend(Protocol):
    """Batch eval and dataset refresh orchestration facade."""

    def trigger_run(self, workflow_id: str, *, parameters: dict[str, str] | None = None) -> WorkflowRunHandle:
        """Start a workflow/deployment run."""

    def poll_status(self, run_id: str) -> WorkflowRunStatus:
        """Fetch current run status."""

    def fetch_logs(self, run_id: str, *, tail_lines: int = 200) -> str:
        """Return recent log output for a run."""

    def list_runs(
        self,
        *,
        workflow_id: str = "",
        limit: int = 20,
    ) -> Sequence[WorkflowRunHandle]:
        """List recent workflow runs, optionally filtered by workflow id."""

    def cancel_run(self, run_id: str) -> WorkflowRunStatus:
        """Request cancellation of a workflow run."""
