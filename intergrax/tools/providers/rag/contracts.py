# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pydantic contracts for ``rag.retrieve`` (Phase O.3)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RagRetrieveInput(BaseModel):
    """LLM-facing input for vector / hybrid document retrieval."""

    query: str = Field(..., min_length=1, description="Natural language search query.")
    top_k: int = Field(default=8, ge=1, le=50, description="Maximum number of chunks to return.")
    session_id: Optional[str] = Field(default=None, description="Scope retrieval to a chat session.")
    user_id: Optional[str] = Field(default=None, description="Scope retrieval to a user.")
    tenant_id: Optional[str] = Field(default=None, description="Scope retrieval to a tenant.")
    workspace_id: Optional[str] = Field(default=None, description="Scope retrieval to a workspace.")
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score; chunks below are dropped.",
    )


class RagChunkResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagRetrieveOutput(BaseModel):
    used: bool
    chunks: list[RagChunkResult] = Field(default_factory=list)
    context_text: str = ""
    reason: str = ""
