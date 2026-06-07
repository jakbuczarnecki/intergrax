# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RagListDocumentsInput(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class RagDocumentSummaryOutput(BaseModel):
    document_id: str


class RagListDocumentsOutput(BaseModel):
    used: bool = False
    documents: list[RagDocumentSummaryOutput] = Field(default_factory=list)
    total: int = 0
    reason: str = ""


class RagGetDocumentInput(BaseModel):
    document_id: str = Field(..., min_length=1)


class RagGetDocumentOutput(BaseModel):
    used: bool = False
    document_id: str = ""
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class RagCheckIndexStatusInput(BaseModel):
    collection: str = Field(default="", description="Optional collection name filter.")


class RagCheckIndexStatusOutput(BaseModel):
    used: bool = False
    ready: bool = False
    collection: str = ""
    document_count: int = 0
    collections: list[str] = Field(default_factory=list)
    reason: str = ""


class RagSearchByMetadataInput(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict, description="Exact-match metadata key/value filters.")
    limit: int = Field(default=50, ge=1, le=500)
    tenant_id: str = Field(default="", description="Optional tenant scope for purge/search.")


class RagMetadataMatchOutput(BaseModel):
    document_id: str
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagSearchByMetadataOutput(BaseModel):
    used: bool = False
    matches: list[RagMetadataMatchOutput] = Field(default_factory=list)
    total: int = 0
    reason: str = ""


class RagPurgeCollectionInput(BaseModel):
    dry_run: bool = Field(default=True, description="When true, report delete count without mutating the index.")
    tenant_id: str = Field(default="", description="Optional tenant scope guard for purge operations.")
    collection: str = Field(default="", description="Optional collection name (informational for multi-collection backends).")


class RagPurgeCollectionOutput(BaseModel):
    used: bool = False
    dry_run: bool = True
    would_delete: int = 0
    deleted: int = 0
    tenant_id: str = ""
    reason: str = ""
