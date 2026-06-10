#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-3.2 — product host intake parity (streaming + durable async)."""

from __future__ import annotations

import sys

from intergrax.applications._shared.intake_wiring import resolve_product_intake_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    wiring = resolve_product_intake_wiring(env)
    if not wiring.durable_async_index:
        print("product intake must enable durable async index", file=sys.stderr)
        return 1
    if not wiring.streaming_intake_enabled:
        print("product intake must enable streaming intake flag", file=sys.stderr)
        return 1

    print("OK: product intake parity (durable async + streaming)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
