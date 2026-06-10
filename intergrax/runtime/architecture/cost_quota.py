# © Artur Czarnecki. All rights reserved.

"""Quota enforcement contracts with deterministic deny/degrade behavior (Phase V-COST.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QuotaResourceType(str, Enum):
    TOKENS = "tokens"
    TOOL_CALLS = "tool_calls"
    RUNTIME_SECONDS = "runtime_seconds"
    CPU_SECONDS = "cpu_seconds"
    MEMORY_MB = "memory_mb"
    CONCURRENT_RUNS = "concurrent_runs"


class QuotaEnforcementAction(str, Enum):
    ALLOW = "allow"
    DEGRADE = "degrade"
    DENY = "deny"


class ResourceQuota(BaseModel):
    resource_type: QuotaResourceType
    scope_id: str
    limit: int
    used: int


class QuotaUsageRequest(BaseModel):
    resource_type: QuotaResourceType
    scope_id: str
    requested_units: int


class QuotaEnforcementDecision(BaseModel):
    action: QuotaEnforcementAction
    allowed_units: int
    reasons: list[str] = Field(default_factory=list)


class QuotaGovernanceReport(BaseModel):
    schema_version: str = "1.0.0"
    quotas: list[ResourceQuota] = Field(default_factory=list)
    decisions: list[QuotaEnforcementDecision] = Field(default_factory=list)


def evaluate_quota_enforcement(
    *,
    quotas: list[ResourceQuota],
    requests: list[QuotaUsageRequest],
    degrade_threshold_ratio: float = 0.90,
) -> QuotaGovernanceReport:
    quota_by_key: dict[tuple[QuotaResourceType, str], ResourceQuota] = {
        (quota.resource_type, quota.scope_id): quota for quota in quotas
    }
    decisions: list[QuotaEnforcementDecision] = []
    for request in requests:
        quota = quota_by_key.get((request.resource_type, request.scope_id))
        if quota is None:
            decisions.append(
                QuotaEnforcementDecision(
                    action=QuotaEnforcementAction.DENY,
                    allowed_units=0,
                    reasons=[f"No quota configured for {request.resource_type.value}:{request.scope_id}"],
                )
            )
            continue
        projected = quota.used + request.requested_units
        if projected > quota.limit:
            decisions.append(
                QuotaEnforcementDecision(
                    action=QuotaEnforcementAction.DENY,
                    allowed_units=0,
                    reasons=[
                        "Quota exceeded: "
                        f"{projected} > {quota.limit} for {request.resource_type.value}:{request.scope_id}"
                    ],
                )
            )
            continue
        usage_ratio = float(projected) / float(max(1, quota.limit))
        if usage_ratio >= degrade_threshold_ratio:
            allowed_units = max(0, quota.limit - quota.used)
            decisions.append(
                QuotaEnforcementDecision(
                    action=QuotaEnforcementAction.DEGRADE,
                    allowed_units=allowed_units,
                    reasons=["Quota near limit; degraded execution path required"],
                )
            )
            continue
        decisions.append(
            QuotaEnforcementDecision(
                action=QuotaEnforcementAction.ALLOW,
                allowed_units=request.requested_units,
                reasons=[],
            )
        )
    return QuotaGovernanceReport(quotas=quotas, decisions=decisions)
