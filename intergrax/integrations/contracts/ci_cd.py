# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CI/CD integration contract (Phase M.6 P4)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


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
    """Read-only CI/CD status facade (GitHub Actions, GitLab CI, …)."""

    def get_workflow_run(self, run_id: str) -> WorkflowRunRecord:
        """Fetch a workflow run by id."""

    def list_check_suites(self, *, ref: str, limit: int = 20) -> Sequence[CheckSuiteRecord]:
        """List recent check suites for a git ref (branch/tag/sha)."""
