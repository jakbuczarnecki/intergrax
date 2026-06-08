# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class ContextSummarizeInput(BaseModel):
    text: str = Field(..., min_length=1)
    max_tokens: int = Field(default=512, ge=32, le=8192)


class ContextSummarizeOutput(BaseModel):
    summary: str
    original_tokens: int = 0
    final_tokens: int = 0
    trimmed: bool = False


class ContextEstimateTokensInput(BaseModel):
    text: str = ""


class ContextEstimateTokensOutput(BaseModel):
    char_count: int = 0
    token_estimate: int = 0
