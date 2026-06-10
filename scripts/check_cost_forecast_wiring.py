#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-24.1 — cost forecasting from historical run patterns."""

from __future__ import annotations

import sys

from intergrax.applications._shared.cost_forecast_wiring import resolve_cost_forecast_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_cost_forecast_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable cost forecasting", file=sys.stderr)
        return 1
    if wiring.report is None or not wiring.report.forecasts:
        print("cost forecast report must include projections", file=sys.stderr)
        return 1

    print(f"OK: cost forecast wiring (forecasts={len(wiring.report.forecasts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
