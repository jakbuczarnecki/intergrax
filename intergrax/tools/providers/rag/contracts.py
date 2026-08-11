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
    allowed_source_ids: tuple[str, ...] | None = Field(
        default=None,
        description="Validated indexed source membership scope (internal retrieval boundary).",
    )


class RagChunkResult(BaseModel):
    id: str
    text: str
    score: float
    rank: int = 0
    channel: str = "unknown"
    vector_id: Optional[str] = None
    scope: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagCitationResult(BaseModel):
    """Structured provenance for a retrieved chunk (M-RAG.29)."""

    chunk_id: str
    source_id: str
    source_type: str = "vectorstore"
    source_label: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None
    excerpt: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagRetrieveOutput(BaseModel):
    used: bool
    chunks: list[RagChunkResult] = Field(default_factory=list)
    citations: list[RagCitationResult] = Field(default_factory=list)
    context_text: str = ""
    reason: str = ""
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieval trace: retriever_id, route_tier, reranker_id, latencies.",
    )
