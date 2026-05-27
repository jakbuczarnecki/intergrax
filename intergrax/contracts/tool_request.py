# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool gateway contracts (architecture §42.12)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.agent_contract_meta import AgentRiskLevel


class ToolResponseStatus(str, Enum):
    SUCCESS = "success"
    DENIED = "denied"
    TIMEOUT = "timeout"
    FAILED = "failed"


class ToolRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"tool_{uuid4().hex[:12]}")
    tool_name: str
    agent_id: str
    step_id: str = ""
    input: Dict[str, Any] = Field(default_factory=dict)
    risk_level: AgentRiskLevel = AgentRiskLevel.LOW
    timeout_ms: int = 30_000
    idempotency_key: Optional[str] = None
    schema_version: str = "tool_request.v1"


class ToolResponse(BaseModel):
    request_id: str
    status: ToolResponseStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: int = 0
    trace_ref: str = ""
    schema_version: str = "tool_response.v1"
