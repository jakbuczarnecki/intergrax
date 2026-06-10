#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-24.3 — CPU/memory/concurrency quotas with tenant fairness."""

from __future__ import annotations

import sys

from intergrax.applications._shared.tenant_quota_wiring import resolve_tenant_quota_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.architecture.cost_quota import QuotaResourceType


def main() -> int:
    wiring = resolve_tenant_quota_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable tenant fairness quotas", file=sys.stderr)
        return 1
    plan = wiring.plan
    if plan is None or not plan.quotas:
        print("tenant fairness quota plan missing", file=sys.stderr)
        return 1
    resource_types = {quota.resource_type for quota in plan.quotas}
    required = {
        QuotaResourceType.CPU_SECONDS,
        QuotaResourceType.MEMORY_MB,
        QuotaResourceType.CONCURRENT_RUNS,
    }
    if not required.issubset(resource_types):
        print("tenant fairness quotas must cover cpu/memory/concurrency", file=sys.stderr)
        return 1
    print(f"OK: tenant fairness quotas ({len(plan.quotas)} tenant quotas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
