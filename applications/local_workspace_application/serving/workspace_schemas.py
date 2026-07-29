# © Artur Czarnecki. All rights reserved.

"""HTTP schemas for managed workspace product API (LKW-PRODUCT-1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreateWorkspaceRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = ""
    tenant_id: str | None = None


class WorkspaceResponseV1(BaseModel):
    workspace_id: str
    tenant_id: str
    name: str
    description: str = ""
    status: str
    created_at: datetime
    updated_at: datetime


class WorkspaceListResponseV1(BaseModel):
    workspaces: list[WorkspaceResponseV1]


class RegisterSourceRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["local_folder"] = "local_folder"
    path: str = Field(..., min_length=1)
    recursive: bool = True


class SourceResponseV1(BaseModel):
    """Detailed source registration response (may include provider locator)."""

    source_id: str
    workspace_id: str
    source_type: str
    path: str
    status: str
    recursive: bool = True
    created_at: datetime | None = None
    last_sync_at: datetime | None = None


class SourceSummaryResponseV1(BaseModel):
    """Safe list-item projection for source inspection (no full locator/path)."""

    source_id: str
    workspace_id: str
    source_type: str
    label: str
    status: str
    recursive: bool = True
    created_at: datetime | None = None
    last_sync_at: datetime | None = None


class SourceListResponseV1(BaseModel):
    sources: list[SourceSummaryResponseV1]


class SyncOperationAcceptedV1(BaseModel):
    operation_id: str
    workspace_id: str
    source_id: str
    status: str


class OperationResponseV1(BaseModel):
    operation_id: str
    operation_type: str
    status: str
    workspace_id: str
    source_id: str
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    documents_indexed: int = 0
    documents_unchanged: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class WorkspaceSearchRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class WorkspaceSearchHitV1(BaseModel):
    document_id: str
    source_id: str
    workspace_id: str
    source_path: str
    file_name: str
    score: float = 0.0
    snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSearchResponseV1(BaseModel):
    workspace_id: str
    query: str
    results: list[WorkspaceSearchHitV1]


class WorkspaceAskRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class WorkspaceAskCitationLocationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int | None = None


class WorkspaceAskCitationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    document_id: str
    source_id: str
    workspace_id: str
    source_path: str
    file_name: str
    excerpt: str = ""
    score: float | None = None
    chunk_id: str | None = None
    location: WorkspaceAskCitationLocationV1 | None = None


class WorkspaceAskErrorV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class WorkspaceAskResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    workspace_id: str
    status: Literal["completed", "insufficient_evidence", "failed"]
    question: str
    answer: str | None = None
    citations: list[WorkspaceAskCitationV1] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None
    error: WorkspaceAskErrorV1 | None = None


class ManagedFileBatchItemAcceptedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: int
    file_name: str
    status: Literal["accepted", "failed"]
    input_id: str | None = None
    source_id: str | None = None
    operation_id: str | None = None
    operation_status: str | None = None
    error_code: str | None = None


class ManagedFileBatchAcceptedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    workspace_id: str
    status: Literal["accepted", "partial", "failed"]
    accepted_count: int
    failed_count: int
    items: list[ManagedFileBatchItemAcceptedV1]


class SourceCandidateSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    label: str
    description: str
    source_type: Literal["local_folder"]
    available: bool


class SourceCandidateListResponseV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    candidates: list[SourceCandidateSummaryV1]


class SourceCandidateAcceptedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    label: str
    workspace_id: str
    source_id: str
    operation_id: str
    status: Literal[
        "accepted",
        "queued",
        "processing",
        "completed",
        "failed",
    ]


class WebUrlIntakeRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., min_length=1, max_length=2048, repr=False)


class WebUrlAcceptedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: str
    workspace_id: str
    source_id: str
    operation_id: str
    status: Literal[
        "accepted",
        "queued",
        "processing",
        "completed",
        "failed",
    ]
    safe_display_url: str
