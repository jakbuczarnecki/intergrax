# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool I/O contracts for codecraft.* (ECC-1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.codecraft.contracts import CraftResult, StaticGateResult


class CodeCraftRunToolInput(BaseModel):
    """Input for ``codecraft.run`` — single-shot gate + sandbox exec."""

    model_config = ConfigDict(extra="forbid")

    code: str
    goal: str | None = None
    language: str = "python"
    timeout_s: float = Field(default=30.0, ge=1.0, le=600.0)
    craft_id: str | None = None
    run_id: str | None = None
    tenant_id: str = "default"
    task_id: str = "default"
    agent_id: str = ""


class CodeCraftRunToolOutput(BaseModel):
    """Output for ``codecraft.run``."""

    model_config = ConfigDict(extra="forbid")

    result: CraftResult
    trace_event_count: int = 0


class CodeCraftDeniedOutput(BaseModel):
    """Fail-closed deny response."""

    model_config = ConfigDict(extra="forbid")

    denied: bool = True
    reason: str
    static_gate: StaticGateResult | None = None
    craft_id: str = ""


__all__ = [
    "CodeCraftDeniedOutput",
    "CodeCraftRunToolInput",
    "CodeCraftRunToolOutput",
    "CraftResult",
    "StaticGateResult",
]
