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
