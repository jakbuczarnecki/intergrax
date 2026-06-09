# © Artur Czarnecki. All rights reserved.

"""Per-tenant cost metrics accumulator (IDEAL-24.2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TenantCostMetrics:
    tenant_id: str
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    run_count: int = 0


@dataclass
class TenantCostMetricsStore:
    _by_tenant: dict[str, TenantCostMetrics] = field(default_factory=dict)

    def record(
        self,
        tenant_id: str,
        *,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> TenantCostMetrics:
        metrics = self._by_tenant.setdefault(tenant_id, TenantCostMetrics(tenant_id=tenant_id))
        metrics.total_tokens += max(tokens, 0)
        metrics.total_cost_usd += max(cost_usd, 0.0)
        metrics.run_count += 1
        return metrics

    def get(self, tenant_id: str) -> TenantCostMetrics | None:
        return self._by_tenant.get(tenant_id)
