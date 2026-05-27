# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox session contracts (architecture §21)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SandboxAuditEntry(BaseModel):
    entry_id: str
    operation: str
    status: str
    started_at_utc: str
    duration_ms: int = 0
    error: Optional[str] = None


class SandboxExecutionResult(BaseModel):
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    audit_entry: Optional[SandboxAuditEntry] = None


class SandboxSessionManifest(BaseModel):
    session_id: str
    tenant_id: str
    task_id: str
    root_path: str
    created_at_utc: str
    allowed_operations: List[str]
    operation_count: int
    cancelled: bool = False
