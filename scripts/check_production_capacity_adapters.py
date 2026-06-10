#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-30.4 — Celery/K8s production-scale adapters."""

from __future__ import annotations

import sys

from intergrax.applications._shared.production_capacity_wiring import resolve_production_capacity_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_production_capacity_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable production capacity adapters", file=sys.stderr)
        return 1
    if wiring.adapters is None or not wiring.probe_passed:
        print("production capacity adapter probe failed", file=sys.stderr)
        return 1
    print("OK: production Celery/K8s capacity adapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
