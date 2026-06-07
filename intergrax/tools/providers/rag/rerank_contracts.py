# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class RagRerankChunkInput(BaseModel):
    id: str = ""
    text: str = Field(..., min_length=1)
    score: float | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RagRerankInput(BaseModel):
    query: str = Field(..., min_length=1)
    chunks: list[RagRerankChunkInput] = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=50)


class RagRerankChunkOutput(BaseModel):
    id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RagRerankOutput(BaseModel):
    query: str
    chunks: list[RagRerankChunkOutput] = Field(default_factory=list)
    reranker_id: str = ""
    total: int = 0
