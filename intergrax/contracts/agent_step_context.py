# © Artur Czarnecki. All rights reserved.

"""Minimal step context for Wave 0 authoring helpers (ACP-STEP-1 expands)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.acp_state import AcpInvocationUsageView
from intergrax.contracts.agent_run_enums import SideEffectMode
from intergrax.contracts.shared_context import SharedContextView


class AgentStepContext(BaseModel):
    """Author-facing step context snapshot (architecture §32.2 target)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    step_index: int = Field(default=0, ge=0)
    run_id: str = ""
    task_id: str = ""
    tenant_id: str = "default"
    workspace_id: str | None = None
    message: str = ""
    step_kind: str | None = None
    agent_id: str = ""
    contract_id: str = ""
    side_effect_mode: SideEffectMode = SideEffectMode.IMMEDIATE
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    llm_router: object | None = Field(default=None, exclude=True, repr=False)
    invocation_usage: AcpInvocationUsageView | None = None
    shared_context: SharedContextView | None = None

    @field_validator("workspace_id")
    @classmethod
    def _workspace_id_non_empty_when_set(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("workspace_id must be non-empty when provided")
        return stripped
