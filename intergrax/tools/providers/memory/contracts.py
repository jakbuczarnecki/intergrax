# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from intergrax.contracts.memory_write_policy import MemoryWritePolicy


class MemoryReadInput(BaseModel):
    namespace: str = Field(..., min_length=1, description="Task memory namespace.")
    key: str = Field(..., min_length=1, description="Record key within the namespace.")


class MemoryReadOutput(BaseModel):
    namespace: str
    key: str
    found: bool
    value: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteInput(BaseModel):
    namespace: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    value: dict[str, Any] = Field(default_factory=dict)
    policy: MemoryWritePolicy = MemoryWritePolicy.REPLACE


class MemoryWriteOutput(BaseModel):
    namespace: str
    key: str
    written: bool = True


class MemoryListKeysInput(BaseModel):
    namespace: str = Field(..., min_length=1)
    prefix: str = ""


class MemoryKeyRecord(BaseModel):
    key: str
    record_id: str = ""
    updated_at_utc: str = ""


class MemoryListKeysOutput(BaseModel):
    namespace: str
    prefix: str
    keys: list[MemoryKeyRecord] = Field(default_factory=list)
    total: int = 0


class MemoryDeleteKeyInput(BaseModel):
    namespace: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)


class MemoryDeleteKeyOutput(BaseModel):
    namespace: str
    key: str
    deleted: bool = False


class MemorySearchInput(BaseModel):
    namespace: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    prefix: str = ""
    limit: int = Field(default=20, ge=1, le=200)


class MemorySearchMatch(BaseModel):
    key: str
    value: dict[str, Any] = Field(default_factory=dict)


class MemorySearchOutput(BaseModel):
    namespace: str
    query: str
    matches: list[MemorySearchMatch] = Field(default_factory=list)
    total: int = 0
