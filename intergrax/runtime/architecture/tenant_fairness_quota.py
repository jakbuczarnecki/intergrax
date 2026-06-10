# © Artur Czarnecki. All rights reserved.

"""Tenant-fair CPU/memory/concurrency quotas (AUDIT-IDEAL-24.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.cost_quota import (
    QuotaGovernanceReport,
    QuotaResourceType,
    QuotaUsageRequest,
    ResourceQuota,
    evaluate_quota_enforcement,
)


class TenantFairnessPool(BaseModel):
    """Shared resource pool distributed fairly across tenants."""

    resource_type: QuotaResourceType
    pool_limit: int = Field(ge=1)
    tenant_ids: list[str] = Field(default_factory=list)


class TenantFairnessQuotaPlan(BaseModel):
    """Per-tenant quotas derived from a shared pool."""

    schema_version: str = "1.0.0"
    quotas: list[ResourceQuota] = Field(default_factory=list)


def build_tenant_fairness_quotas(pools: list[TenantFairnessPool]) -> TenantFairnessQuotaPlan:
    """Split each pool evenly across tenants with deterministic integer allocation."""
    quotas: list[ResourceQuota] = []
    for pool in pools:
        tenants = sorted(set(pool.tenant_ids))
        if not tenants:
            continue
        base = pool.pool_limit // len(tenants)
        remainder = pool.pool_limit % len(tenants)
        for index, tenant_id in enumerate(tenants):
            limit = base + (1 if index < remainder else 0)
            quotas.append(
                ResourceQuota(
                    resource_type=pool.resource_type,
                    scope_id=tenant_id,
                    limit=limit,
                    used=0,
                )
            )
    return TenantFairnessQuotaPlan(quotas=quotas)


def evaluate_tenant_fairness_enforcement(
    *,
    pools: list[TenantFairnessPool],
    requests: list[QuotaUsageRequest],
    degrade_threshold_ratio: float = 0.90,
) -> QuotaGovernanceReport:
    """Evaluate requests against tenant-fair quotas."""
    plan = build_tenant_fairness_quotas(pools)
    return evaluate_quota_enforcement(
        quotas=plan.quotas,
        requests=requests,
        degrade_threshold_ratio=degrade_threshold_ratio,
    )
