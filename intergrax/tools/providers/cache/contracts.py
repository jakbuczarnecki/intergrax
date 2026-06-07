# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import base64

from pydantic import BaseModel, Field, field_validator


class CacheGetInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)


class CacheGetOutput(BaseModel):
    tenant_id: str
    key: str
    found: bool
    value_base64: str = ""


class CacheSetInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    value_base64: str = Field(..., min_length=1)
    ttl_seconds: int | None = Field(default=None, ge=1, le=86400)

    @field_validator("value_base64")
    @classmethod
    def _validate_base64(cls, value: str) -> str:
        base64.b64decode(value, validate=True)
        return value


class CacheSetOutput(BaseModel):
    tenant_id: str
    key: str
    stored: bool = True


class CacheDeleteInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)


class CacheDeleteOutput(BaseModel):
    tenant_id: str
    key: str
    deleted: bool = True


class CacheListKeysInput(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    prefix: str = ""
    limit: int = Field(default=100, ge=1, le=1000)


class CacheListKeysOutput(BaseModel):
    tenant_id: str
    keys: list[str] = Field(default_factory=list)
    total: int = 0
