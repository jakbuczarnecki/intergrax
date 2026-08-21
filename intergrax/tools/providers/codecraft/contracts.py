# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool I/O contracts for codecraft.* (ECC-1+)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.codecraft.contracts import CodeCraftSession, CraftResult


class CodeCraftContextFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    tenant_id: str = "default"
    task_id: str = "default"
    agent_id: str = ""


class CodeCraftRunToolInput(CodeCraftContextFields):
    code: str
    goal: str | None = None
    language: str = "python"
    timeout_s: float = Field(default=30.0, ge=1.0, le=600.0)
    craft_id: str | None = None


class CodeCraftStartToolInput(CodeCraftContextFields):
    goal: str
    constraints: str = ""
    language: str = "python"
    craft_id: str | None = None
    initial_code: str | None = None


class CodeCraftIterateToolInput(CodeCraftContextFields):
    craft_id: str
    patch_diagnostics: str = ""
    timeout_s: float = Field(default=30.0, ge=1.0, le=600.0)


class CodeCraftGetStateToolInput(CodeCraftContextFields):
    craft_id: str


class CodeCraftDisposeToolInput(CodeCraftContextFields):
    craft_id: str


class CodeCraftPromoteToolInput(CodeCraftContextFields):
    craft_id: str


class CodeCraftListEphemeralToolsInput(CodeCraftContextFields):
    craft_id: str


class CodeCraftRunToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: CraftResult
    trace_event_count: int = 0


class CodeCraftStartToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: CodeCraftSession | None = None
    error: str = ""
    trace_event_count: int = 0


class CodeCraftIterateToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: CodeCraftSession | None = None
    result: CraftResult
    trace_event_count: int = 0


class CodeCraftGetStateToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: CodeCraftSession | None = None
    found: bool = False


class CodeCraftDisposeToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposed: bool = False
    craft_id: str = ""
    trace_event_count: int = 0


class CodeCraftPromoteToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: CraftResult


class CodeCraftListEphemeralToolsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    craft_id: str
    tool_ids: list[str] = Field(default_factory=list)
