# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentParseInput(BaseModel):
    source_path: str = Field(..., min_length=1, description="Local filesystem path to parse.")


class DocumentFragmentOutput(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentParseOutput(BaseModel):
    parser_id: str
    fragments: list[DocumentFragmentOutput] = Field(default_factory=list)
    fragment_count: int = 0


class DocumentParsePreviewInput(BaseModel):
    source_path: str = Field(..., min_length=1, description="Local filesystem path to parse.")
    max_fragments: int = Field(default=5, ge=1, le=50)
    max_chars_per_fragment: int = Field(default=2000, ge=100, le=20000)


class DocumentParsePreviewOutput(BaseModel):
    parser_id: str
    fragments: list[DocumentFragmentOutput] = Field(default_factory=list)
    fragment_count: int = 0
    truncated: bool = False
