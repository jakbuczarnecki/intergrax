#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-AHI.3 — capability marketplace readiness gate."""

from __future__ import annotations

import sys

from intergrax.applications._shared.capability_marketplace_wiring import resolve_capability_marketplace_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_capability_marketplace_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable capability marketplace readiness", file=sys.stderr)
        return 1
    report = wiring.report
    if report is None or not report.ready:
        print("capability marketplace readiness checks failed", file=sys.stderr)
        return 1
    print("OK: capability marketplace readiness (trust/certification/billing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
