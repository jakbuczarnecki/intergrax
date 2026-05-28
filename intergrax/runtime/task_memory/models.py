# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical task-scoped memory records (§27, Phase I.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskMemoryRecord(BaseModel):
    """
    Single namespaced key/value entry owned by Nexus for one task.

    Namespaces group related facts (``vendor_report``, ``research``, …).
    """

    record_id: str = Field(default_factory=lambda: f"tm_{uuid4().hex}")
    tenant_id: str
    task_id: str
    namespace: str
    key: str
    value: Dict[str, Any] = Field(default_factory=dict)
    created_at_utc: str = Field(default_factory=utc_now_iso)
    updated_at_utc: str = Field(default_factory=utc_now_iso)
    schema_version: str = "task_memory.v1"
    provenance: Dict[str, Any] = Field(default_factory=dict)

    @property
    def qualified_key(self) -> str:
        return f"{self.namespace}/{self.key}"


class TaskMemoryWriteRequest(BaseModel):
    namespace: str
    key: str
    value: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class TaskMemoryQuery(BaseModel):
    tenant_id: str
    task_id: str
    namespace: str
    key: Optional[str] = None
    prefix: str = ""
    limit: int = 100
