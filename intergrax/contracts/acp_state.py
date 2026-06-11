# © Artur Czarnecki. All rights reserved.

"""ACP session state envelope (architecture §32.0 · ACP-0)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run_enums import CognitivePattern

ACP_STATE_KEY = "acp.state.v1"
ACP_STATE_SCHEMA_VERSION: Literal["acp.state.v1"] = "acp.state.v1"
ACP_USAGE_KEY = "acp.usage.v1"


class AcpTokenUsage(BaseModel):
    """LLM token rollup for one scope (architecture §25.4 · §25.5 · ACP-TOK-1)."""

    model_config = ConfigDict(extra="forbid")

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    tokens_total: int = Field(default=0, ge=0)
    tokens_limit: int | None = Field(default=None, ge=1)
    tokens_remaining: int | None = Field(default=None, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class AcpInvocationUsageView(BaseModel):
    """
    Read-only invocation usage for ``on_next_step`` (architecture §25.4).

    ``agent`` mirrors ``acp.state.v1.budget`` token fields; ``environment`` is the
    task/application rollup (Nexus graph or host session).
    """

    model_config = ConfigDict(extra="forbid")

    agent: AcpTokenUsage = Field(default_factory=AcpTokenUsage)
    environment: AcpTokenUsage = Field(default_factory=AcpTokenUsage)


class AcpBudgetState(BaseModel):
    """Budget counters tracked inside session state (architecture §25.2 · TOOL-ENG-6)."""

    model_config = ConfigDict(extra="forbid")

    steps_used: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    tokens_total: int = Field(default=0, ge=0)
    tokens_limit: int | None = Field(default=None, ge=1)
    tokens_remaining: int | None = Field(default=None, ge=0)
    cost_usd: float = 0.0
    react_iterations_used: int = Field(default=0, ge=0)
    react_iterations_max: int = Field(default=0, ge=0)


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
