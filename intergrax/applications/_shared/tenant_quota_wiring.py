# © Artur Czarnecki. All rights reserved.

"""Tenant-fair CPU/memory/concurrency quota wiring (AUDIT-IDEAL-24.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.cost_quota import QuotaResourceType
from intergrax.runtime.architecture.tenant_fairness_quota import (
    TenantFairnessPool,
    TenantFairnessQuotaPlan,
    build_tenant_fairness_quotas,
)


@dataclass(frozen=True, slots=True)
class TenantQuotaWiring:
    enabled: bool
    plan: TenantFairnessQuotaPlan | None


def resolve_tenant_quota_wiring(env: ApplicationEnvironmentProfile) -> TenantQuotaWiring:
    """Build tenant-fair resource quotas for product hosts."""
    cost = env.cost_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return TenantQuotaWiring(enabled=False, plan=None)
    if not cost.tenant_fairness_quotas_enabled:
        return TenantQuotaWiring(enabled=False, plan=None)

    tenant_ids = ["tenant_a", "tenant_b", "tenant_c"]
    pools = [
        TenantFairnessPool(
            resource_type=QuotaResourceType.CPU_SECONDS,
            pool_limit=cost.max_tool_calls or 64,
            tenant_ids=tenant_ids,
        ),
        TenantFairnessPool(
            resource_type=QuotaResourceType.MEMORY_MB,
            pool_limit=cost.max_total_tokens or 32_000,
            tenant_ids=tenant_ids,
        ),
        TenantFairnessPool(
            resource_type=QuotaResourceType.CONCURRENT_RUNS,
            pool_limit=cost.max_llm_calls or 32,
            tenant_ids=tenant_ids,
        ),
    ]
    plan = build_tenant_fairness_quotas(pools)
    return TenantQuotaWiring(enabled=True, plan=plan)
