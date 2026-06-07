# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class WebsearchInvalidateCacheInput(BaseModel):
    query: str = Field(default="", description="Optional query prefix to invalidate.")
    clear_all: bool = Field(default=False, description="When true, clear the entire query cache.")


class WebsearchInvalidateCacheOutput(BaseModel):
    used: bool = False
    invalidated: int = 0
    reason: str = ""
