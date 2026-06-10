# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RagIngestInput(BaseModel):
    source_path: str = Field(..., min_length=1, description="Local filesystem path to ingest.")
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagIngestOutput(BaseModel):
    used: bool
    num_chunks: int = 0
    vector_ids: list[str] = Field(default_factory=list)
    parser_id: Optional[str] = None
    parser_trace: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    file_size_bytes: int = 0
    async_job_recommended: bool = False


class RagScheduleIngestJobInput(BaseModel):
    source_path: str = Field(..., min_length=1, description="Local filesystem path to ingest asynchronously.")
    workflow_id: Optional[str] = Field(
        default=None,
        description="Orchestrator workflow/deployment id; defaults to RagProfile.async_ingest_workflow_id.",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional stable key; when omitted a deterministic key is derived from path + scope.",
    )
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagScheduleIngestJobOutput(BaseModel):
    used: bool
    run_id: str = ""
    status: str = ""
    url: str = ""
    idempotency_key: str = ""
    reason: str = ""
