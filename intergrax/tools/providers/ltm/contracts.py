# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class LtmSearchInput(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)


class LtmMemoryHit(BaseModel):
    entry_id: str
    content: str
    kind: str = ""
    score: float = 0.0


class LtmSearchOutput(BaseModel):
    used: bool = False
    hits: list[LtmMemoryHit] = Field(default_factory=list)
    reason: str = ""


class LtmWriteFactInput(BaseModel):
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    kind: str = Field(default="user_fact", description="MemoryKind value.")
    title: str = ""


class LtmWriteFactOutput(BaseModel):
    written: bool = False
    entry_id: str = ""
