#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-30.4 — Celery/K8s production-scale adapters."""

from __future__ import annotations

import sys

from intergrax.applications._shared.production_capacity_governance_wiring import (
    build_production_capacity_governance,
)
from intergrax.applications._shared.production_capacity_wiring import resolve_production_capacity_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    governance = build_production_capacity_governance(env)
    wiring = resolve_production_capacity_wiring(env, governance=governance)
    if not wiring.enabled:
        print("product host must enable production capacity adapters", file=sys.stderr)
        return 1
    if governance.mutation_authorization_boundary is None:
        if wiring.adapters is not None or wiring.probe_passed:
            print(
                "production capacity wiring must fail closed without canonical policy",
                file=sys.stderr,
            )
            return 1
        print(
            "OK: production Celery/K8s capacity adapters "
            "(canonical policy required for governed probe)",
        )
        return 0
    if wiring.adapters is None or not wiring.probe_passed:
        print("production capacity adapter probe failed", file=sys.stderr)
        return 1
    print("OK: production Celery/K8s capacity adapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
