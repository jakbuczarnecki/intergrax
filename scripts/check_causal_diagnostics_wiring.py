#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-21.1 — causal diagnostics beyond trace bridge."""

from __future__ import annotations

import sys

from intergrax.applications._shared.causal_diagnostics_wiring import resolve_causal_diagnostics_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_causal_diagnostics_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable causal diagnostics", file=sys.stderr)
        return 1
    if wiring.chain is None or not wiring.chain.links:
        print("causal chain must include diagnostic links", file=sys.stderr)
        return 1
    print(f"OK: causal diagnostics wiring ({len(wiring.chain.links)} links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
