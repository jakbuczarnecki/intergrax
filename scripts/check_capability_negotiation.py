#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-19.2 — capability negotiation at runtime resolve."""

from __future__ import annotations

import sys

from intergrax.applications._shared.capability_negotiation_wiring import negotiate_runtime_capabilities
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    result = negotiate_runtime_capabilities(
        ("echo", "missing.capability"),
        available_capabilities=("echo", "legal.research"),
        env=env,
    )
    if result.granted != ("echo",):
        print("negotiation must grant only available capabilities", file=sys.stderr)
        return 1
    if not result.denied:
        print("negotiation must deny unknown capabilities", file=sys.stderr)
        return 1
    if result.negotiated:
        print("product host must fail negotiation when capabilities are denied", file=sys.stderr)
        return 1

    lab = negotiate_runtime_capabilities(
        ("echo",),
        available_capabilities=("echo",),
        env=ApplicationEnvironmentProfile.lab_defaults(),
    )
    if not lab.negotiated:
        print("lab host negotiation must succeed for available capabilities", file=sys.stderr)
        return 1

    print("OK: capability negotiation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
