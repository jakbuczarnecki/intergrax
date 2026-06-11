# © Artur Czarnecki. All rights reserved.

"""Tool execution metadata for mutability and compensation (architecture §40.3 · ACP-PROD-3)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel


class ToolMutability(StrEnum):
    READ_ONLY = "read_only"
    MUTATING = "mutating"


class ToolReversibility(StrEnum):
    NONE = "none"
    COMPENSATABLE = "compensatable"
    MANUAL = "manual"


class ToolExecutionProfile(BaseModel):
    """Harness classification for tool invoke policy."""

    model_config = ConfigDict(extra="forbid")

    tool_id: str
    mutability: ToolMutability = ToolMutability.READ_ONLY
    reversibility: ToolReversibility = ToolReversibility.NONE
    requires_approval: bool = False
    supports_dry_run: bool = False
    requires_idempotency_key: bool = False
    compensation_tool_id: str | None = None
    max_retry: int = Field(default=1, ge=1)
    timeout_ms: int = Field(default=30_000, ge=1)


def profile_from_tool_contract(contract: ToolContract) -> ToolExecutionProfile:
    """Derive execution profile from catalog ``ToolContract`` metadata."""
    mutating = contract.side_effects
    requires_hitl = contract.risk_level in {ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL}
    return ToolExecutionProfile(
        tool_id=contract.tool_id,
        mutability=ToolMutability.MUTATING if mutating else ToolMutability.READ_ONLY,
        reversibility=ToolReversibility.MANUAL if mutating else ToolReversibility.NONE,
        requires_approval=requires_hitl,
        supports_dry_run=not mutating,
        requires_idempotency_key=mutating,
        max_retry=contract.retry_policy.max_attempts,
        timeout_ms=contract.timeout_ms,
    )


def build_profile_map(contracts: list[ToolContract]) -> dict[str, ToolExecutionProfile]:
    return {contract.tool_id: profile_from_tool_contract(contract) for contract in contracts}
