#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-6.2 — live cost/latency/quality model routing (AHI prod path)."""

from __future__ import annotations

import sys

from intergrax.applications._shared.llm_routing_wiring import resolve_live_model_routing_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_live_model_routing_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable live model routing", file=sys.stderr)
        return 1
    if wiring.routing_decision is None:
        print("model routing decision missing", file=sys.stderr)
        return 1
    if wiring.engine_id != "routing_tuning":
        print("routing engine id mismatch", file=sys.stderr)
        return 1
    print(f"OK: live model routing ({wiring.routing_decision.routing_reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
