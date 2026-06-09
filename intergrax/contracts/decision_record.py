# © Artur Czarnecki. All rights reserved.

"""Universal decision record for UAEP and Nexus steps (FAUDIT-COG.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DecisionRecord(BaseModel):
    """Typed rationale artifact for model/tool/subagent choices."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: f"dec_{uuid4().hex}")
    trace_id: str = ""
    run_id: str = ""
    tenant_id: str = ""
    task_id: str = ""
    agent_id: str = ""
    step_id: str = ""
    decision_type: str = ""
    rationale: str = ""
    policy_action: str = ""
    delegation_target: str = ""
    delegation_rationale: str = ""
    delegation_scopes: tuple[str, ...] = ()
    version: str = "decision_record.v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
