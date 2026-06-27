#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-16.1 — online context drift monitoring gate."""

from __future__ import annotations

import sys

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.context.context_drift_monitor import (
    ContextDriftSignal,
    evaluate_context_drift,
)


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    if not env.context_profile.drift_monitoring_enabled:
        print("product_defaults must enable drift_monitoring", file=sys.stderr)
        return 1

    stable = evaluate_context_drift(
        ContextDriftSignal(token_estimate=1000, chunk_count=4, baseline_token_estimate=1000),
        alert_threshold=env.context_profile.drift_alert_threshold,
    )
    if stable.alert:
        print("stable context must not alert", file=sys.stderr)
        return 1

    drifted = evaluate_context_drift(
        ContextDriftSignal(token_estimate=1600, chunk_count=2, baseline_token_estimate=1000),
        alert_threshold=env.context_profile.drift_alert_threshold,
    )
    if not drifted.alert:
        print("drifted context must alert", file=sys.stderr)
        return 1

    print("OK: context drift monitoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
