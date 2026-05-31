# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class RagListCollectionsInput(BaseModel):
    pass


class RagListCollectionsOutput(BaseModel):
    used: bool = False
    collections: list[str] = Field(default_factory=list)
    reason: str = ""
