# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class RagDeleteDocumentsInput(BaseModel):
    document_ids: list[str] = Field(..., min_length=1, description="Vector chunk/document ids to delete.")


class RagDeleteDocumentsOutput(BaseModel):
    used: bool = False
    deleted_count: int = 0
    reason: str = ""


class RagDescribeCollectionInput(BaseModel):
    collection: str = Field(
        default="",
        description="Optional collection name; empty uses the active/default collection.",
    )


class RagDescribeCollectionOutput(BaseModel):
    used: bool = False
    collection: str = ""
    document_count: int = 0
    collections: list[str] = Field(default_factory=list)
    reason: str = ""
