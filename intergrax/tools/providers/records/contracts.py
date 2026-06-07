# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RecordsGetInput(BaseModel):
    partition_key: str = Field(..., min_length=1)
    row_key: str = Field(..., min_length=1)


class RecordsDocumentOutput(BaseModel):
    partition_key: str
    row_key: str
    data: dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: Optional[int] = None


class RecordsGetOutput(BaseModel):
    found: bool = False
    document: Optional[RecordsDocumentOutput] = None


class RecordsPutInput(BaseModel):
    partition_key: str = Field(..., min_length=1)
    row_key: str = Field(..., min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: Optional[int] = None


class RecordsPutOutput(BaseModel):
    stored: bool = True
    partition_key: str
    row_key: str


class RecordsDeleteInput(BaseModel):
    partition_key: str = Field(..., min_length=1)
    row_key: str = Field(..., min_length=1)


class RecordsDeleteOutput(BaseModel):
    deleted: bool = True
    partition_key: str
    row_key: str


class RecordsQueryInput(BaseModel):
    partition_key: str = Field(..., min_length=1)
    limit: int = Field(default=100, ge=1, le=1000)
    row_key_prefix: Optional[str] = None


class RecordsQueryOutput(BaseModel):
    documents: list[RecordsDocumentOutput] = Field(default_factory=list)
    total: int = 0


class RecordsDescribeCollectionInput(BaseModel):
    partition_key: str = Field(..., min_length=1)
    sample_limit: int = Field(default=5, ge=1, le=50)


class RecordsDescribeCollectionOutput(BaseModel):
    used: bool = False
    partition_key: str = ""
    document_count: int = 0
    sample_row_keys: list[str] = Field(default_factory=list)
    sample_field_names: list[str] = Field(default_factory=list)
    reason: str = ""
