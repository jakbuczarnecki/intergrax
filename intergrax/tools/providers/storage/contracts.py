# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import base64

from pydantic import BaseModel, Field, field_validator


class StorageGetInput(BaseModel):
    key: str = Field(..., min_length=1)


class StorageGetOutput(BaseModel):
    key: str
    found: bool
    body_base64: str = ""
    content_type: str = "application/octet-stream"
    size_bytes: int = 0


class StoragePutInput(BaseModel):
    key: str = Field(..., min_length=1)
    body_base64: str = Field(..., min_length=1, description="Base64-encoded object bytes.")
    content_type: str = "application/octet-stream"

    @field_validator("body_base64")
    @classmethod
    def _validate_base64(cls, value: str) -> str:
        base64.b64decode(value, validate=True)
        return value


class StoragePutOutput(BaseModel):
    key: str
    stored: bool = True
    size_bytes: int = 0


class StoragePresignedUrlInput(BaseModel):
    key: str = Field(..., min_length=1)
    expires_in_seconds: int = Field(default=3600, ge=60, le=86400)
    method: str = Field(default="GET", pattern="^(GET|PUT)$")


class StoragePresignedUrlOutput(BaseModel):
    key: str
    url: str
    method: str
    expires_in_seconds: int


class StorageDeleteInput(BaseModel):
    key: str = Field(..., min_length=1)


class StorageDeleteOutput(BaseModel):
    key: str
    deleted: bool = True
