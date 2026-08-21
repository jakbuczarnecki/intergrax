# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ECC domain contracts — sessions, gate results, promotion (ECC-1+)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CraftVerdict = Literal["continue", "revise", "promote", "abort"]
CraftSessionStatus = Literal["open", "pending_hitl", "closed", "disposed"]


class StaticGateResult(BaseModel):
    """Outcome of L0 static analysis before sandbox execution."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    rule_ids: list[str] = Field(default_factory=list)
    message: str = ""


class IterationRecord(BaseModel):
    """One craft loop iteration snapshot."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    static_gate: StaticGateResult
    exec_success: bool = False
    test_passed: bool | None = None
    verdict: CraftVerdict = "continue"
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


class CodeCraftSession(BaseModel):
    """In-memory craft session state (ECC-2)."""

    model_config = ConfigDict(extra="forbid")

    craft_id: str
    task_id: str
    tenant_id: str
    run_id: str | None = None
    goal: str
    mode: str
    language: str = "python"
    code: str = ""
    status: CraftSessionStatus = "open"
    iteration: int = 0
    max_iterations: int = 8
    total_exec_time_s: float = 0.0
    pending_hitl: bool = False
    hitl_approved: bool = False
    promoted: bool = False
    structured_output: dict[str, Any] = Field(default_factory=dict)
    iterations: list[IterationRecord] = Field(default_factory=list)
    ephemeral_tool_ids: list[str] = Field(default_factory=list)
    sandbox_session_id: str | None = None
    error: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    disposed: bool = False


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
