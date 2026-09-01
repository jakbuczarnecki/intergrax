# © Artur Czarnecki. All rights reserved.

"""Shared contracts for TOOLS-SIDE-EFFECT-SAFETY Docker proof."""

from __future__ import annotations

from pydantic import BaseModel, Field

PROOF_TOOL_CHARGE = "proof.side_effect.charge"
PROOF_TOOL_CHARGE_ALT = "proof.side_effect.charge_alt"
PROOF_TOOL_FAIL_BEFORE = "proof.side_effect.fail_before"
PROOF_TOOL_BAD_OUTPUT = "proof.side_effect.bad_output"
PROOF_TOOL_SLOW_CHARGE = "proof.side_effect.slow_charge"
PROOF_GOVERNANCE_TOOL = "proof.side_effect.governance"
DEFAULT_TENANT = "proof-tenant"
DEFAULT_AGENT = "proof-agent"


class ChargeInput(BaseModel):
    business_operation_id: str = Field(min_length=1)
    amount: int = Field(default=100, ge=0)
    proof_mode: str = "normal"
    proof_delay_ms: int = 0
    http_timeout_s: float = 120.0


class ChargeOutput(BaseModel):
    effect_id: int
    business_operation_id: str
    committed_at: str


class InvokeRequest(BaseModel):
    run_id: str
    step_id: str = "step1"
    tool_id: str = PROOF_TOOL_CHARGE
    business_operation_id: str
    amount: int = 100
    idempotency_key: str
    tenant_id: str = DEFAULT_TENANT
    proof_mode: str = "normal"
    proof_delay_ms: int = 0
    worker_source: str | None = None
    governance_action: str | None = None
    governance_rule_id: str = "proof.governance.rule"
    require_hitl: bool = False
    hitl_resume: bool = False
    timeout_ms: int | None = None


class InvokeResponse(BaseModel):
    success: bool
    replay: bool = False
    blocked: bool = False
    uncertain: bool = False
    error_type: str | None = None
    error_code: str | None = None
    ledger_status: str | None = None
    output: dict | None = None
