# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CI/CD integration contract (Phase M.6 P4)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from intergrax.integrations.contracts.base import HealthStatus


class WorkflowRunRecord(BaseModel):
    id: str
    name: str = ""
    status: str = ""
    conclusion: str = ""
    url: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class CheckSuiteRecord(BaseModel):
    id: str
    name: str = ""
    status: str = ""
    conclusion: str = ""
    url: str = ""


@runtime_checkable
class CiCdBackend(Protocol):
    """CI/CD status and workflow run operations (GitHub Actions, GitLab CI, …)."""

    def get_workflow_run(self, run_id: str) -> WorkflowRunRecord:
        """Fetch a workflow run by id."""

    def list_check_suites(self, *, ref: str, limit: int = 20) -> Sequence[CheckSuiteRecord]:
        """List recent check suites for a git ref (branch/tag/sha)."""

    def list_workflow_runs(
        self,
        *,
        workflow_id: str = "",
        ref: str = "",
        limit: int = 20,
    ) -> Sequence[WorkflowRunRecord]:
        """List recent workflow runs, optionally filtered by workflow id or git ref."""

    def cancel_workflow_run(self, run_id: str) -> WorkflowRunRecord:
        """Request cancellation of a workflow run and return updated status."""

    def health(self) -> HealthStatus | bool:
        """Optional startup / connectivity probe."""
