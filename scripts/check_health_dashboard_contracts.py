#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-21.2 — quality / governance / cost health dashboard contracts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.health_dashboard_wiring import resolve_health_dashboard_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_health_dashboard_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable health dashboard contracts", file=sys.stderr)
        return 1
    contract = wiring.contract
    if contract is None:
        print("health dashboard contract missing", file=sys.stderr)
        return 1
    if contract.governance.tenant_isolation_verified is not True:
        print("governance health must verify tenant isolation on product hosts", file=sys.stderr)
        return 1
    print("OK: health dashboard contracts (quality/governance/cost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
