# © Artur Czarnecki. All rights reserved.

"""Domain models for Trusted Ask Workspace (MVP-2)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1


class AskRunStatus(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class AskAnswerAssemblyStatus(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AskAnswerAssemblyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AskAnswerAssemblyStatus
    answer: str | None = None
    used_evidence_ids: list[str] = Field(default_factory=list)


class AskCitationLocation(BaseModel):
    """Approved location metadata only — no provider/Slack leakage."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = None


class AskCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)
    excerpt: str = ""
    score: float | None = None
    chunk_id: str | None = None
    location: AskCitationLocation | None = None


class AskError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class WorkspaceAskRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    status: AskRunStatus
    evidence: list[WorkspaceSearchHitV1] = Field(default_factory=list)
    answer: str | None = None
    citations: list[AskCitation] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None
    error: AskError | None = None


class AskAnswerAssemblyError(RuntimeError):
    """Answer assembly failed (parse, validation, or unknown evidence IDs)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
