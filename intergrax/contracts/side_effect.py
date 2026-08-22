# © Artur Czarnecki. All rights reserved.

"""Side-effect ledger contracts (architecture §40.2 · ACP-PROD-2)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SideEffectKind(StrEnum):
    TOOL = "tool"
    RAG_WRITE = "rag_write"
    LLM_CACHE_WRITE = "llm_cache_write"
    ARTIFACT_PUBLISH = "artifact_publish"


class SideEffectStatus(StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"
    FAILED = "failed"
    COMPENSATED = "compensated"


class SideEffectRecord(BaseModel):
    """One attempted external side effect within an agent run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["side_effect.v1"] = "side_effect.v1"
    side_effect_id: str
    idempotency_key: str
    run_id: str
    step_index: int
    kind: SideEffectKind = SideEffectKind.TOOL
    target: str
    status: SideEffectStatus = SideEffectStatus.PENDING
    committed_at: datetime | None = None
    external_ref: str | None = None
    committed_externally: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompensationRequest(BaseModel):
    """Compensation invoke for a committed side effect (architecture §40.3.3 · ACP-PROD-3)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["compensation_request.v1"] = "compensation_request.v1"
    original_side_effect_id: str
    compensation_tool_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str


class AgentRunCheckpoint(BaseModel):
    """Durable step checkpoint for ACP session resume (architecture §40.1 · ACP-PROD-1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_run_checkpoint.v1"] = "agent_run_checkpoint.v1"
    run_id: str
    tenant_id: str
    agent_id: str
    step_index: int = Field(ge=0)
    revision: int = Field(default=1, ge=1)
    state_root: dict[str, Any] = Field(default_factory=dict)
    side_effect_ledger: list[SideEffectRecord] = Field(default_factory=list)
    trace_step_count: int = Field(default=0, ge=0)
    saved_at: datetime = Field(default_factory=_utc_now)
