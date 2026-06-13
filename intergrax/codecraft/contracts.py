# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ECC domain contracts — sessions, gate results, promotion (ECC-1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CraftVerdict = Literal["continue", "revise", "promote", "abort"]


class StaticGateResult(BaseModel):
    """Outcome of L0 static analysis before sandbox execution."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    rule_ids: list[str] = Field(default_factory=list)
    message: str = ""


class CraftResult(BaseModel):
    """Typed promotion contract for a craft operation (ECC-1 single-shot)."""

    model_config = ConfigDict(extra="forbid")

    craft_id: str
    success: bool
    mode: str
    static_gate: StaticGateResult
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    sandbox_session_id: str | None = None
    error: str = ""
    structured_output: dict[str, Any] = Field(default_factory=dict)
    verdict: CraftVerdict = "abort"


class CodeCraftRunInput(BaseModel):
    """Internal service input for single-shot craft."""

    model_config = ConfigDict(extra="forbid")

    code: str
    goal: str | None = None
    language: str = "python"
    timeout_s: float = Field(default=30.0, ge=1.0, le=600.0)
    craft_id: str | None = None
