# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class CostGetRunBudgetInput(BaseModel):
    tenant_id: str = Field(default="", description="Optional tenant scope for budget envelope lookup.")


class CostGetRunBudgetOutput(BaseModel):
    configured: bool
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_llm_calls: int | None = None
    max_tool_calls: int | None = None
    max_wall_time_seconds: float | None = None
    within_budget: bool = True
    remaining_amount: float | None = None


class CostCheckQuotaInput(BaseModel):
    resource_type: str = Field(..., min_length=1, description="tokens | tool_calls | runtime_seconds")
    scope_id: str = Field(..., min_length=1)
    requested_units: int = Field(..., ge=1)


class CostCheckQuotaOutput(BaseModel):
    action: str
    allowed_units: int = 0
    reasons: list[str] = Field(default_factory=list)
