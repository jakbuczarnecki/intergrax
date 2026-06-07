# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class VectorStoreCountInput(BaseModel):
    tenant_id: str = Field(default="default")


class VectorStoreCountOutput(BaseModel):
    used: bool = False
    document_count: int = 0
    reason: str = ""


class VectorStoreDeleteInput(BaseModel):
    tenant_id: str = Field(default="default")
    document_ids: list[str] = Field(..., min_length=1, max_length=200)


class VectorStoreDeleteOutput(BaseModel):
    used: bool = False
    deleted_count: int = 0
    reason: str = ""


class VectorStoreListCollectionsInput(BaseModel):
    tenant_id: str = Field(default="default")


class VectorStoreListCollectionsOutput(BaseModel):
    used: bool = False
    collections: list[str] = Field(default_factory=list)
    reason: str = ""


class VectorStoreHealthInput(BaseModel):
    tenant_id: str = Field(default="default")


class VectorStoreHealthOutput(BaseModel):
    used: bool = False
    healthy: bool = False
    document_count: int = 0
    reason: str = ""
