# © Artur Czarnecki. All rights reserved.

"""ACP session state envelope (architecture §32.0 · ACP-0)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_enums import CognitivePattern

ACP_STATE_KEY = "acp.state.v1"
ACP_STATE_SCHEMA_VERSION: Literal["acp.state.v1"] = "acp.state.v1"


class AcpBudgetState(BaseModel):
    """Budget counters tracked inside session state."""

    model_config = ConfigDict(extra="forbid")

    steps_used: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0


class AcpSessionState(BaseModel):
    """
    Platform session state envelope — authors subclass with extra=forbid.

    Serialized under ``ACP_STATE_KEY`` inside run request/result state blobs.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["acp.state.v1"] = ACP_STATE_SCHEMA_VERSION
    state_version: int = Field(default=0, ge=0, alias="_version")
    pattern: CognitivePattern | None = None
    phase: str | None = None
    iteration: int = Field(default=0, ge=0)
    budget: AcpBudgetState | None = None
