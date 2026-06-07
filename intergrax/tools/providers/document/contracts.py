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
