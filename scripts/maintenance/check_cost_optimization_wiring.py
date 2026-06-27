#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-24.2 — automated cost optimization recommendations."""

from __future__ import annotations

import sys

from intergrax.applications._shared.cost_optimization_wiring import resolve_cost_optimization_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_cost_optimization_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable cost optimization recommendations", file=sys.stderr)
        return 1
    if wiring.report is None or not wiring.report.recommendations:
        print("optimization report must include recommendations", file=sys.stderr)
        return 1
    if not all(item.policy_compliant for item in wiring.report.recommendations):
        print("optimization recommendations must be policy compliant", file=sys.stderr)
        return 1

    print(f"OK: cost optimization wiring ({len(wiring.report.recommendations)} recommendations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
