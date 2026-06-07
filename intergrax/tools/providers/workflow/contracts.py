# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Workflow orchestrator tool I/O contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorkflowTriggerInput(BaseModel):
    workflow_id: str
    parameters: dict[str, str] = Field(default_factory=dict)


class WorkflowTriggerOutput(BaseModel):
    run_id: str
    status: str = "pending"
    url: str = ""


class WorkflowPollInput(BaseModel):
    run_id: str


class WorkflowPollOutput(BaseModel):
    run_id: str
    status: str = ""
    conclusion: str = ""
    logs_uri: str = ""


class WorkflowFetchLogsInput(BaseModel):
    run_id: str
    tail_lines: int = Field(default=200, ge=1, le=5000)


class WorkflowFetchLogsOutput(BaseModel):
    run_id: str
    logs: str = ""


class WorkflowListRunsInput(BaseModel):
    workflow_id: str = Field(default="", description="Optional workflow identifier filter.")
    limit: int = Field(default=20, ge=1, le=100)


class WorkflowRunSummaryOutput(BaseModel):
    run_id: str
    status: str = ""
    url: str = ""


class WorkflowListRunsOutput(BaseModel):
    runs: list[WorkflowRunSummaryOutput] = Field(default_factory=list)
    total: int = 0


class WorkflowCancelRunInput(BaseModel):
    run_id: str = Field(..., min_length=1)
